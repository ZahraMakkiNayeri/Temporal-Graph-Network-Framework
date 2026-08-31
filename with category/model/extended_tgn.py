from model.tgn import TGN

# =====================================================================
# extendedtgn.py  —  REVISED (Reviewer 6, Comments 1d and 1e)
#
# Fixes in this cell:
#   (1d) DOUBLE SIGMOID removed. The parent TGN.compute_edge_probabilities()
#        already returns sigmoid-transformed scores; the previous code
#        applied sigmoid a second time, squeezing every probability into
#        (0.5, 0.731) and making every edge positive at the 0.5 threshold
#        (the degenerate Recall=1.0, Acc=Prec=0.5 signature in Fig. 19).
#        We now compute the raw affinity scores once and apply exactly
#        ONE sigmoid.
#   (1e) DOUBLE FORWARD PASS removed. The previous code called
#        super().compute_edge_probabilities() and then called
#        compute_temporal_embeddings() a second time for the category
#        head. With memory enabled that (i) updated the memory twice per
#        batch and (ii) let the current edge's own message — stored by
#        the first pass — leak into its own category prediction via
#        get_updated_memory() in the second pass. Both heads now share
#        ONE call to compute_temporal_embeddings(), so memory is
#        updated exactly once per batch and category predictions use
#        memory as of the PREVIOUS batches only.
#   (1c) Inputs are normalized to numpy arrays, because the parent TGN
#        implementation expects numpy (np.concatenate / torch.from_numpy);
#        the released evaluation code passed torch tensors, which cannot
#        execute when memory is enabled.
# =====================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_numpy(x):
    """Accept numpy arrays, lists, or torch tensors and return numpy."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class ExtendedTGN(TGN):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2, n_heads=2, dropout=0.1,
                 use_memory=False, memory_update_at_start=True, message_dimension=100, memory_dimension=500,
                 embedding_module_type="graph_attention", message_function="mlp", mean_time_shift_src=0,
                 std_time_shift_src=1, mean_time_shift_dst=0, std_time_shift_dst=1, n_neighbors=None,
                 aggregator_type="bigru_transformer", memory_updater_type="gru",
                 use_destination_embedding_in_message=False, use_source_embedding_in_message=False,
                 dyrep=False, num_categories=10,
                 aggregator_hidden_dim=64, aggregator_n_layers=1,
                 tcn_channels=None, tcn_kernel_size=3, aggregator_max_seq_len=128):
        super(ExtendedTGN, self).__init__(
            neighbor_finder=neighbor_finder, node_features=node_features, edge_features=edge_features,
            device=device, n_layers=n_layers, n_heads=n_heads, dropout=dropout, use_memory=use_memory,
            memory_update_at_start=memory_update_at_start, message_dimension=message_dimension,
            memory_dimension=memory_dimension, embedding_module_type=embedding_module_type,
            message_function=message_function, mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src, mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst, n_neighbors=n_neighbors,
            aggregator_type=aggregator_type, memory_updater_type=memory_updater_type,
            use_destination_embedding_in_message=use_destination_embedding_in_message,
            use_source_embedding_in_message=use_source_embedding_in_message, dyrep=dyrep,
            aggregator_hidden_dim=aggregator_hidden_dim, aggregator_n_layers=aggregator_n_layers,
            tcn_channels=tcn_channels, tcn_kernel_size=tcn_kernel_size,
            aggregator_max_seq_len=aggregator_max_seq_len)

        # Linear layer for edge-category prediction
        self.category_predictor = nn.Linear(self.embedding_dimension, num_categories)

    def compute_edge_probabilities_and_categories(self, source_nodes, destination_nodes, negative_nodes,
                                                  edge_times, edge_idxs, n_neighbors=20):
        """
        Compute link probabilities for positive/negative edges AND category logits
        for the positive edges, from a SINGLE forward pass (single memory update).
        """
        source_nodes = _as_numpy(source_nodes)
        destination_nodes = _as_numpy(destination_nodes)
        negative_nodes = _as_numpy(negative_nodes)
        edge_times = _as_numpy(edge_times).astype(np.float64)
        edge_idxs = _as_numpy(edge_idxs)

        n_samples = len(source_nodes)

        # ONE forward pass: embeddings for sources, destinations, negatives.
        # Memory (if enabled) is updated exactly once, inside this call.
        source_node_embedding, destination_node_embedding, negative_node_embedding = \
            self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes,
                                             edge_times, edge_idxs, n_neighbors)

        # Link scores (same decoder as the parent class), ONE sigmoid only.
        score = self.affinity_score(
            torch.cat([source_node_embedding, source_node_embedding], dim=0),
            torch.cat([destination_node_embedding, negative_node_embedding])).squeeze(dim=0)
        pos_prob = score[:n_samples].sigmoid()
        neg_prob = score[n_samples:].sigmoid()

        # Stash the source embeddings of this batch for the link-ranking
        # evaluation (Reviewer 6, Comment 5). Detached: only read at eval time.
        self._last_source_embeddings = source_node_embedding.detach()

        # Category logits from the SAME embeddings (no second forward pass).
        combined_embeddings = source_node_embedding + destination_node_embedding
        category_logits = self.category_predictor(combined_embeddings)

        return pos_prob, neg_prob, category_logits

    def score_candidate_destinations(self, candidate_nodes, times, n_neighbors=20):
        """FIX (Reviewer 6, Comment 5): scores for GENUINE link ranking.

        For each positive edge i of the batch most recently passed through
        compute_edge_probabilities_and_categories (whose source embeddings were
        stashed), compute the decoder score of source_i against each candidate
        destination candidate_nodes[i, q], embedded AT the positive edge's
        timestamp from the CURRENT persisted memory state. The pending raw
        messages (this batch's own edges) are exactly what must stay invisible,
        and Memory.get_memory excludes them by construction.

        candidate_nodes: (n, Q) numpy int array; times: (n,) numpy array.
        Returns raw decoder scores of shape (n, Q).
        """
        candidate_nodes = np.asarray(candidate_nodes)
        times = np.asarray(times, dtype=np.float64)
        n, Q = candidate_nodes.shape
        flat_nodes = candidate_nodes.reshape(-1)
        flat_times = np.repeat(times, Q)

        memory = None
        time_diffs = None
        if self.use_memory:
            memory = self.memory.get_memory(list(range(self.n_nodes)))
            last_update = self.memory.last_update
            flat_times_t = torch.from_numpy(flat_times).float().to(self.device)
            # same normalization as the parent's negative/destination branch
            time_diffs = flat_times_t - last_update[flat_nodes].float()
            time_diffs = (time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

        candidate_embeddings = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=flat_nodes,
            timestamps=flat_times,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs)

        source_rep = self._last_source_embeddings.repeat_interleave(Q, dim=0)
        scores = self.affinity_score(source_rep, candidate_embeddings).squeeze(dim=0)
        return scores.view(n, Q)
