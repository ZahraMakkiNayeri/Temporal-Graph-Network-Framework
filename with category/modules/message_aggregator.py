# =====================================================================
# message_aggregator.py  —  REVISED (Reviewer 6, Comment 1b)
#
# Fixes in this cell:
#   (1b) aggregate() of all learnable aggregators now uses the SAME
#        interface as LastMessageAggregator / MeanMessageAggregator:
#        aggregate(node_ids, messages) where `messages` is a
#        defaultdict {node_id: [(message_tensor, timestamp), ...]}.
#        The previous code called self.group_by_id(node_ids,
#        messages[0], messages[1]), which indexed the message lists of
#        NODES 0 and 1 instead of unpacking (messages, timestamps) —
#        this is why enabling memory crashed.
#   (1b) The missing TCN class is now defined (causal temporal
#        convolutional network with Chomp1d), so TCNBlockAggregator
#        no longer references an undefined name.
#   (1b) Every learnable aggregator now consumes AND emits vectors of
#        `input_dim` (= raw_message_dimension in tgn.py), via explicit
#        input/output projections, so the aggregated message is
#        compatible with the message function and memory updater
#        regardless of hidden sizes / head counts.
#   Sequences are padded and processed as a batch (pad_sequence +
#   src_key_padding_mask / pack_padded_sequence) instead of one
#   node at a time, which also improves runtime.
# =====================================================================

from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a dict mapping node_id -> list of (message, timestamp)
    tuples, aggregate the messages of each node with one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: defaultdict {node_id: [(message_tensor, timestamp_tensor), ...]}
    :return: (to_update_node_ids, aggregated_messages, last_timestamps)
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages

  def _collect_sequences(self, node_ids, messages):
    """Walk the TGN message store (like LastMessageAggregator) and return each
    node's chronologically sorted message history as a list of tensors.

    MEMORY FIX (first real-data run, OOM on T4): sequences are truncated to the
    most recent `max_seq_len` messages — a bounded attention window over each
    node's pending history. This is a disclosed model hyperparameter
    (config: aggregator_max_seq_len)."""
    unique_node_ids = np.unique(node_ids)
    to_update_node_ids, sequences, last_timestamps = [], [], []

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        sorted_msgs = sorted(messages[node_id], key=lambda m: float(m[1]))
        if self.max_seq_len is not None and len(sorted_msgs) > self.max_seq_len:
          sorted_msgs = sorted_msgs[-self.max_seq_len:]
        to_update_node_ids.append(node_id)
        sequences.append(torch.stack([m[0] for m in sorted_msgs]))
        last_timestamps.append(sorted_msgs[-1][1])

    return to_update_node_ids, sequences, last_timestamps

  def _iter_padded_buckets(self, ids, sequences, last_timestamps,
                           token_budget=16384):
    """MEMORY FIX: yield length-sorted buckets instead of padding every pending
    node's sequence to one global L_max.

    On real alert data, thousands of nodes hold 1-2 pending messages while a
    few flood victims hold hundreds; global padding makes the BiGRU/Transformer
    activations scale as N_nodes x L_max (mostly padding) — the source of the
    CUDA OOM on the first real-data run. Sorting by length and capping each
    bucket at ~token_budget padded cells makes the work scale with the SUM of
    true sequence lengths instead.

    Yields (order_index, padded, padding_mask, lengths) per bucket; callers
    scatter their per-bucket outputs back through order_index."""
    order = sorted(range(len(sequences)), key=lambda k: sequences[k].shape[0], reverse=True)
    start = 0
    while start < len(order):
      L = sequences[order[start]].shape[0]
      width = max(1, token_budget // max(L, 1))
      chunk = order[start:start + width]
      seqs = [sequences[k] for k in chunk]
      lengths = torch.tensor([sq.shape[0] for sq in seqs], dtype=torch.int64)
      padded = pad_sequence(seqs, batch_first=True).to(self.device)
      padding_mask = (torch.arange(padded.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)).to(self.device)
      yield chunk, padded, padding_mask, lengths
      start += width

  def _aggregate_bucketed(self, node_ids, messages):
    """Shared aggregate() implementation for the learnable aggregators:
    collect -> bucket -> per-bucket _encode_padded -> reassemble in order."""
    ids, sequences, last_ts = self._collect_sequences(node_ids, messages)
    if len(ids) == 0:
      return [], torch.empty(0), torch.empty(0)

    outputs = [None] * len(sequences)
    for chunk, padded, padding_mask, lengths in self._iter_padded_buckets(ids, sequences, last_ts):
      encoded = self._encode_padded(padded, padding_mask, lengths)   # (len(chunk), input_dim)
      for row, k in enumerate(chunk):
        outputs[k] = encoded[row]

    return ids, torch.stack(outputs), torch.stack(last_ts)

  @staticmethod
  def _masked_mean(x, padding_mask):
    """Mean over the sequence dimension, ignoring PAD positions.
    x: (N, L, D); padding_mask: (N, L) True at PAD."""
    valid = (~padding_mask).unsqueeze(-1).float()                           # (N, L, 1)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Average all pending messages for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class TemporalPositionalEncoding(nn.Module):
    """(Moved verbatim from the standalone cell so this file is self-contained
    when exported as modules/message_aggregator.py — Reviewer 6, 1c.)"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# ---------------------------------------------------------------------
# TCN — previously referenced by TCNBlockAggregator but never defined
# (Reviewer 6, 1b: "the TCN aggregator references an undefined class")
# ---------------------------------------------------------------------
class Chomp1d(nn.Module):
    """Removes the trailing padding so the convolution stays causal."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Causal temporal convolutional network.
    Input/output shape: (batch, seq_len, features)."""
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation,
                                        padding=(kernel_size - 1) * dilation,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # (B, T, D) -> (B, D, T) for Conv1d, back to (B, T, C)
        return self.network(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------
# BiTA — the proposed aggregator (BiGRU + Transformer encoder)
# ---------------------------------------------------------------------
class BiGRUTransformerAggregator(MessageAggregator):
    def __init__(self, input_dim, hidden_dim, n_heads, dropout, device, max_seq_len=None):
        super(BiGRUTransformerAggregator, self).__init__(device)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.bigru = nn.GRU(input_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=2 * hidden_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

        # FIX (1b): project back to input_dim so the aggregated message has the
        # dimensionality expected by the message function / memory updater.
        self.output_proj = nn.Linear(2 * hidden_dim, input_dim)

    def aggregate(self, node_ids, messages):
        return self._aggregate_bucketed(node_ids, messages)

    def _encode_padded(self, padded, padding_mask, lengths):
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.bigru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)         # (n, L, 2H)
        transformer_out = self.transformer(gru_out, src_key_padding_mask=padding_mask)
        return self.output_proj(self._masked_mean(transformer_out, padding_mask))



# ---------------------------------------------------------------------
# Recurrent-only aggregators — needed to reproduce Table 9 / Experiment 10
# (GRU, BiGRU, Transformer, GRU+Transformer, BiGRU+Transformer). Without
# these the marginal contribution of the Transformer stage, which Reviewers
# 10 #2, 8 #1 and 6 #9 all ask about, cannot be measured.
# ---------------------------------------------------------------------
class GRUAggregator(MessageAggregator):
    """Unidirectional GRU over the message sequence, mean-pooled readout."""
    def __init__(self, input_dim, hidden_dim, dropout, device, bidirectional=False,
                 max_seq_len=None):
        super().__init__(device)
        self.max_seq_len = max_seq_len
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1,
                          bidirectional=bidirectional, batch_first=True)
        self.output_proj = nn.Linear((2 if bidirectional else 1) * hidden_dim, input_dim)

    def aggregate(self, node_ids, messages):
        return self._aggregate_bucketed(node_ids, messages)

    def _encode_padded(self, padded, padding_mask, lengths):
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.output_proj(self._masked_mean(out, padding_mask))


class BiGRUAggregator(GRUAggregator):
    """BiGRU without the Transformer stage — the direct ablation of BiTA."""
    def __init__(self, input_dim, hidden_dim, dropout, device, max_seq_len=None):
        super().__init__(input_dim, hidden_dim, dropout, device,
                         bidirectional=True, max_seq_len=max_seq_len)


class GRUTransformerAggregator(MessageAggregator):
    """Unidirectional GRU followed by a Transformer encoder: isolates the
    contribution of BIDIRECTIONALITY when compared against BiTA."""
    def __init__(self, input_dim, hidden_dim, n_heads, dropout, device, max_seq_len=None):
        super().__init__(device)
        self.max_seq_len = max_seq_len
        d_model = n_heads * math.ceil(hidden_dim / n_heads)
        self.gru = nn.GRU(input_dim, d_model, num_layers=1,
                          bidirectional=False, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, input_dim)

    def aggregate(self, node_ids, messages):
        return self._aggregate_bucketed(node_ids, messages)

    def _encode_padded(self, padded, padding_mask, lengths):
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.transformer(out, src_key_padding_mask=padding_mask)
        return self.output_proj(self._masked_mean(out, padding_mask))


class _ProjectedTransformerAggregator(MessageAggregator):
    """Shared base for the Transformer-only aggregator variants.

    FIX (1b): nn.TransformerEncoderLayer requires n_heads to divide d_model,
    but input_dim (= raw_message_dimension) is data-dependent and generally
    not divisible by n_heads. We therefore project input -> d_model -> input,
    with d_model the smallest multiple of n_heads >= input_dim.
    """
    def __init__(self, input_dim, n_heads, n_layers, dropout, device, max_seq_len=None):
        super().__init__(device)
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        d_model = n_heads * math.ceil(input_dim / n_heads)
        self.d_model = d_model
        self.in_proj = nn.Identity() if d_model == input_dim else nn.Linear(input_dim, d_model)
        self.out_proj = nn.Identity() if d_model == input_dim else nn.Linear(d_model, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _encode(self, x, padding_mask):
        return self.encoder(x, src_key_padding_mask=padding_mask)

    def aggregate(self, node_ids, messages):
        return self._aggregate_bucketed(node_ids, messages)

    def _encode_padded(self, padded, padding_mask, lengths):
        x = self.in_proj(padded)
        encoded = self._encode(x, padding_mask)
        return self.out_proj(self._masked_mean(encoded, padding_mask))


class BiTransformerAggregator(_ProjectedTransformerAggregator):
    pass


class BiTransformerTemporalAggregator(_ProjectedTransformerAggregator):
    def __init__(self, input_dim, n_heads, n_layers, dropout, device, max_seq_len=None):
        super().__init__(input_dim, n_heads, n_layers, dropout, device, max_seq_len=max_seq_len)
        self.temporal_encoding = TemporalPositionalEncoding(
            d_model=self.d_model, dropout=dropout, max_len=max(500, max_seq_len or 0))

    def _encode(self, x, padding_mask):
        return self.encoder(self.temporal_encoding(x), src_key_padding_mask=padding_mask)


class RelativeTransformerAggregator(_ProjectedTransformerAggregator):
    def __init__(self, input_dim, n_heads, n_layers, dropout, device, max_seq_len=None):
        super().__init__(input_dim, n_heads, n_layers, dropout, device, max_seq_len=max_seq_len)
        self.pos_encoder = nn.Parameter(torch.randn(max(500, max_seq_len or 0), self.d_model))

    def _encode(self, x, padding_mask):
        seq_len = x.shape[1]
        return self.encoder(x + self.pos_encoder[:seq_len].unsqueeze(0),
                            src_key_padding_mask=padding_mask)


class StackedBiTransformerAggregator(_ProjectedTransformerAggregator):
    def __init__(self, input_dim, n_heads, n_layers, dropout, device, max_seq_len=None):
        super().__init__(input_dim, n_heads, n_layers, dropout, device, max_seq_len=max_seq_len)
        self.residual_proj = nn.Linear(self.d_model, self.d_model)

    def _encode_padded(self, padded, padding_mask, lengths):
        x = self.in_proj(padded)
        residual = self.residual_proj(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.out_proj(self._masked_mean(encoded + residual, padding_mask))


class TCNBlockAggregator(MessageAggregator):
    def __init__(self, input_dim, num_channels, kernel_size, dropout, device, max_seq_len=None):
        super().__init__(device)
        self.max_seq_len = max_seq_len
        self.tcn = TCN(input_dim, num_channels, kernel_size, dropout)
        # FIX (1b): project the last TCN channel size back to input_dim.
        self.output_proj = nn.Linear(num_channels[-1], input_dim)

    def aggregate(self, node_ids, messages):
        return self._aggregate_bucketed(node_ids, messages)

    def _encode_padded(self, padded, padding_mask, lengths):
        output = self.tcn(padded)                                            # (n, L, C), causal
        idx = (lengths - 1).to(self.device)
        last_step = output[torch.arange(output.size(0), device=self.device), idx]
        return self.output_proj(last_step)


def get_message_aggregator(
    aggregator_type,
    device,
    input_dim=None,
    hidden_dim=None,
    n_heads=None,
    dropout=None,
    n_layers=1,
    tcn_channels=None,
    tcn_kernel_size=None,
    max_seq_len=None
):
    if aggregator_type == "last":
        return LastMessageAggregator(device=device)
    elif aggregator_type == "mean":
        return MeanMessageAggregator(device=device)
    elif aggregator_type == "bigru_transformer":
        if None in [input_dim, hidden_dim, n_heads, dropout]:
            raise ValueError("BiGRUTransformerAggregator requires input_dim, hidden_dim, n_heads, and dropout")
        return BiGRUTransformerAggregator(input_dim, hidden_dim, n_heads, dropout, device, max_seq_len=max_seq_len)
    elif aggregator_type == "gru":
        if None in [input_dim, hidden_dim, dropout]:
            raise ValueError("GRUAggregator requires input_dim, hidden_dim and dropout")
        return GRUAggregator(input_dim, hidden_dim, dropout, device, bidirectional=False,
                             max_seq_len=max_seq_len)
    elif aggregator_type == "bigru":
        if None in [input_dim, hidden_dim, dropout]:
            raise ValueError("BiGRUAggregator requires input_dim, hidden_dim and dropout")
        return BiGRUAggregator(input_dim, hidden_dim, dropout, device, max_seq_len=max_seq_len)
    elif aggregator_type == "gru_transformer":
        if None in [input_dim, hidden_dim, n_heads, dropout]:
            raise ValueError("GRUTransformerAggregator requires input_dim, hidden_dim, n_heads, dropout")
        return GRUTransformerAggregator(input_dim, hidden_dim, n_heads, dropout, device,
                                        max_seq_len=max_seq_len)
    elif aggregator_type == "bitransformer":
        if None in [input_dim, n_heads, n_layers, dropout]:
            raise ValueError("BiTransformerAggregator requires input_dim, n_heads, n_layers, and dropout")
        return BiTransformerAggregator(input_dim, n_heads, n_layers, dropout, device, max_seq_len=max_seq_len)
    elif aggregator_type == "bitransformer_temporal":
        if None in [input_dim, n_heads, n_layers, dropout]:
            raise ValueError("BiTransformerTemporalAggregator requires input_dim, n_heads, n_layers, and dropout")
        return BiTransformerTemporalAggregator(input_dim, n_heads, n_layers, dropout, device, max_seq_len=max_seq_len)
    elif aggregator_type == "relative_transformer":
        if None in [input_dim, n_heads, n_layers, dropout]:
            raise ValueError("RelativeTransformerAggregator requires input_dim, n_heads, n_layers, and dropout")
        return RelativeTransformerAggregator(input_dim, n_heads, n_layers, dropout, device, max_seq_len=max_seq_len)
    elif aggregator_type == "stacked_bitransformer":
        if None in [input_dim, n_heads, n_layers, dropout]:
            raise ValueError("StackedBiTransformerAggregator requires input_dim, n_heads, n_layers, and dropout")
        return StackedBiTransformerAggregator(input_dim, n_heads, n_layers, dropout, device, max_seq_len=max_seq_len)
    elif aggregator_type == "tcn":
        if None in [input_dim, tcn_channels, tcn_kernel_size, dropout]:
            raise ValueError("TCNBlockAggregator requires input_dim, tcn_channels (list), kernel_size, and dropout")
        return TCNBlockAggregator(input_dim, tcn_channels, tcn_kernel_size, dropout, device, max_seq_len=max_seq_len)
    else:
        raise ValueError(f"Message aggregator '{aggregator_type}' not implemented")
