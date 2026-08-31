# =====================================================================
# evaluation.py  —  REVISED (Reviewer 6, Comments 1c, 1d, 1e)
#
# Fixes in this cell:
#   (1c) The model is called with numpy arrays (as in training) instead of
#        torch tensors; the tensor path could not execute with memory
#        enabled (torch.from_numpy on a tensor in get_raw_messages).
#   (1d) The raw link scores and labels are now RETURNED (inside the
#        `link_extras` dict at tuple position 6, previously a dict holding
#        only edge_loss) so that threshold-based link metrics can be
#        computed with an explicitly disclosed, validation-tuned threshold
#        instead of the distorted default-0.5 protocol. Threshold-based
#        link metrics at `link_threshold` are also reported directly.
#   The returned tuple keeps its original length (22) and ordering, so
#   downstream unpacking is unchanged.
# =====================================================================

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

@torch.no_grad()
def eval_edge_prediction_with_categories(
    model,
    negative_edge_sampler,
    data,
    n_neighbors=20,
    device='cpu',
    edge_criterion=None,
    category_criterion=None,
    link_threshold=0.5,
    link_ranking_candidates=0,
    ranking_seed=123,
    candidate_destinations=None,
    seq_length_bins=False,
    streaming=False,
    batch_size=200
):
    """seq_length_bins: record each edge's pending-message count so metrics can be
       reported per message-sequence-length bin (Experiment 8).
       streaming: evaluate one interaction at a time (batch_size=1), the true
       online setting requested by Reviewer 6 Comment 8. Slow but exact."""
    model.eval()

    if streaming:
        batch_size = 1
    num_instances = len(data.sources)
    num_batches = (num_instances + batch_size - 1) // batch_size

    all_pos_scores, all_neg_scores = [], []
    all_true_labels, all_pred_labels = [], []
    all_category_logits = []

    total_edge_loss = 0.0
    total_category_loss = 0.0

    # FIX (Reviewer 6, Comment 5): genuine link-ranking state. When
    # link_ranking_candidates = Q > 0, each positive edge's destination is
    # ranked against Q sampled candidate destinations, scored by the same
    # decoder at the same timestamp from the same memory state.
    all_seq_lengths = []          # pending messages of each positive edge's source
    rank_rng = np.random.RandomState(ranking_seed)
    if link_ranking_candidates and candidate_destinations is None:
        # Uniform over DISTINCT candidate nodes. Passing a raw destination
        # array would draw duplicates, counting one competitor many times.
        candidate_destinations = np.unique(data.destinations)
    all_link_ranks = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(num_instances, start_idx + batch_size)

        sources_batch = data.sources[start_idx:end_idx]
        destinations_batch = data.destinations[start_idx:end_idx]
        edge_idxs_batch = data.edge_idxs[start_idx:end_idx]
        timestamps_batch = data.timestamps[start_idx:end_idx]
        categories_batch = data.labels[start_idx:end_idx]

        size = len(sources_batch)

        # Experiment 8: how many messages BiTA actually aggregates for this edge.
        # Recorded BEFORE prediction, so it reflects the state the model sees.
        if seq_length_bins:
            if getattr(model, 'use_memory', False):
                store = model.memory.messages
                all_seq_lengths.extend(len(store[int(s)]) for s in sources_batch)
            else:
                all_seq_lengths.extend([0] * size)
        # FIX (Reviewer 6, Comment 2): historical/inductive/collision-checked
        # samplers need the current batch's positive edges.
        if getattr(negative_edge_sampler, 'needs_batch_context', False):
            _, negatives_batch = negative_edge_sampler.sample(
                size, sources=sources_batch, destinations=destinations_batch)
        else:
            _, negatives_batch = negative_edge_sampler.sample(size)

        # FIX (1c): pass numpy arrays, matching the training loop and the
        # parent TGN implementation's expectations.
        pos_score, neg_score, category_logits = model.compute_edge_probabilities_and_categories(
            sources_batch,
            destinations_batch,
            negatives_batch,
            timestamps_batch,
            edge_idxs_batch,
            n_neighbors=n_neighbors
        )

        pred_categories = torch.argmax(category_logits, dim=1).cpu().numpy()
        all_true_labels.extend(categories_batch)
        all_pred_labels.extend(pred_categories)
        all_category_logits.append(category_logits.cpu().numpy())
        all_pos_scores.append(pos_score.squeeze(-1).cpu().numpy() if pos_score.dim() > 1 else pos_score.cpu().numpy())
        all_neg_scores.append(neg_score.squeeze(-1).cpu().numpy() if neg_score.dim() > 1 else neg_score.cpu().numpy())

        # FIX (Reviewer 6, Comment 5): GENUINE link ranking.
        if link_ranking_candidates:
            Q = link_ranking_candidates
            cand = rank_rng.choice(candidate_destinations, size=(size, Q))
            for _ in range(3):  # keep the true destination out of its own candidate set
                clash = cand == np.asarray(destinations_batch)[:, None]
                if not clash.any():
                    break
                cand[clash] = rank_rng.choice(candidate_destinations, size=int(clash.sum()))
            cand_probs = model.score_candidate_destinations(
                cand, timestamps_batch, n_neighbors=n_neighbors).sigmoid()
            _pos = pos_score.squeeze(-1) if pos_score.dim() > 1 else pos_score
            greater = (cand_probs > _pos.unsqueeze(1)).sum(dim=1).float()
            ties = (cand_probs == _pos.unsqueeze(1)).sum(dim=1).float()
            all_link_ranks.append((1.0 + greater + 0.5 * ties).cpu().numpy())

        # Edge loss (probabilities carry exactly ONE sigmoid now — fix 1d)
        if edge_criterion is not None:
            pos_label = torch.ones_like(pos_score.squeeze(), device=pos_score.device)
            neg_label = torch.zeros_like(neg_score.squeeze(), device=neg_score.device)
            edge_loss = edge_criterion(pos_score.squeeze(), pos_label) + edge_criterion(neg_score.squeeze(), neg_label)
            total_edge_loss += edge_loss.item()

        # Category loss
        if category_criterion is not None:
            true_labels_tensor = torch.tensor(categories_batch, dtype=torch.long, device=category_logits.device)
            cat_loss = category_criterion(category_logits, true_labels_tensor)
            total_category_loss += cat_loss.item()

    # Aggregate
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_pred_labels)
    category_logits = np.concatenate(all_category_logits)
    pos_scores = np.concatenate(all_pos_scores)
    neg_scores = np.concatenate(all_neg_scores)

    y_scores = torch.softmax(torch.tensor(category_logits), dim=1).numpy()

    # Binary classification (link prediction) — rank-based metrics
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    auc_score = roc_auc_score(all_labels, all_scores)
    avg_precision = average_precision_score(all_labels, all_scores)

    # FIX (Reviewer 6, Comment 5): aggregate the genuine link-ranking metrics.
    link_ranking = None
    if link_ranking_candidates and all_link_ranks:
        _ranks = np.concatenate(all_link_ranks)
        link_ranking = {
            'num_candidates': link_ranking_candidates,
            'mrr': float(np.mean(1.0 / _ranks)),
            'hits@1': float(np.mean(_ranks <= 1)),
            'hits@3': float(np.mean(_ranks <= 3)),
            'hits@10': float(np.mean(_ranks <= 10)),
            'protocol': ('each positive ranked against {} uniformly sampled candidate '
                         'destinations (true destination excluded), scored at the positive '
                         'edge timestamp from the persisted memory state; ties count 0.5'
                         ).format(link_ranking_candidates),
        }

    # ---- Experiment 8: metrics per message-sequence-length bin ----------
    seq_bins = None
    if seq_length_bins and all_seq_lengths:
        lens = np.array(all_seq_lengths)
        edges = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9)]
        names = ['1', '2-5', '6-10', '11-20', '21-50', '>50']
        seq_bins = {}
        for (lo_b, hi_b), nm in zip(edges, names):
            m = (lens >= lo_b) & (lens <= hi_b)
            if m.sum() < 2:
                continue
            b_scores = np.concatenate([pos_scores[m], neg_scores[m]])
            b_labels = np.concatenate([np.ones(int(m.sum())), np.zeros(int(m.sum()))])
            entry = {'n': int(m.sum()),
                     'link_f1': f1_score(b_labels, (b_scores >= link_threshold).astype(int),
                                         zero_division=0),
                     'link_acc': accuracy_score(b_labels, (b_scores >= link_threshold).astype(int)),
                     'category_acc': accuracy_score(y_true[m], y_pred[m])}
            if len(np.unique(b_labels)) > 1:
                entry['auc'] = roc_auc_score(b_labels, b_scores)
                entry['ap'] = average_precision_score(b_labels, b_scores)
            seq_bins[nm] = entry

    # FIX (1d): threshold-based link metrics computed at an EXPLICIT threshold
    # (default 0.5; pass a validation-tuned value for the disclosed protocol),
    # and the raw scores/labels returned for external threshold tuning.
    link_pred = (all_scores >= link_threshold).astype(int)
    link_extras = {
        'edge_loss': total_edge_loss / num_batches if edge_criterion else None,
        'pos_scores': pos_scores,
        'neg_scores': neg_scores,
        'link_labels': all_labels,
        'link_threshold': link_threshold,
        'link_accuracy': accuracy_score(all_labels, link_pred),
        'link_precision': precision_score(all_labels, link_pred, zero_division=0),
        'link_recall': recall_score(all_labels, link_pred, zero_division=0),
        'link_f1': f1_score(all_labels, link_pred, zero_division=0),
        'link_ranking': link_ranking,   # FIX (Comment 5): genuine link-ranking metrics (None if disabled)
        'seq_length_bins': seq_bins,    # Experiment 8 (None if disabled)
        'link_pr_curve': precision_recall_curve(all_labels, all_scores),   # R1-minor 5
        'link_roc_curve': roc_curve(all_labels, all_scores),
    }

    # CATEGORY Hits@K — computed over the 4-class category softmax.
    # FIX (Reviewer 6, Comment 5): these are CLASSIFICATION metrics (Hits@1 is
    # identically accuracy; random baseline: Hits@3 = 0.75 with 4 classes).
    # They must be reported as such — genuine link ranking lives in
    # link_extras['link_ranking'].
    hits_at_k = lambda k: np.mean([
        y_true[i] in np.argsort(-y_scores[i])[:k] for i in range(len(y_true))
    ])
    hits_at_1 = hits_at_k(1)
    hits_at_3 = hits_at_k(3)
    hits_at_5 = hits_at_k(5)

    # CATEGORY MRR — same caveat (random baseline ~= 0.521 with 4 classes).
    ranks = np.argsort(-y_scores, axis=1)
    correct_ranks = np.array([np.where(ranks[i] == y_true[i])[0][0] + 1 for i in range(len(y_true))])
    mrr = np.mean(1 / correct_ranks)

    # Overall classification metrics
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Confusion matrix for FPR/FNR/TPR/TNR
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(category_logits.shape[1]))
    # R1-minor 5/6: expose the confusion matrix and category scores so the
    # confusion-matrix figure and per-class PR curves can be regenerated.
    link_extras['category_confusion_matrix'] = cm.copy()
    link_extras['category_scores'] = y_scores
    link_extras['category_true'] = y_true
    fpr, fnr, tpr, tnr = {}, {}, {}, {}
    TP_total, FP_total, FN_total, TN_total = 0, 0, 0, 0
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        TP_total += TP
        FP_total += FP
        FN_total += FN
        TN_total += TN

        fpr[i] = FP / (FP + TN + 1e-9)
        fnr[i] = FN / (FN + TP + 1e-9)
        tpr[i] = TP / (TP + FN + 1e-9)
        tnr[i] = TN / (TN + FP + 1e-9)

    # Global (macro) confusion-based metrics
    fpr_macro = FP_total / (FP_total + TN_total + 1e-9)
    fnr_macro = FN_total / (FN_total + TP_total + 1e-9)
    tpr_macro = TP_total / (TP_total + FN_total + 1e-9)
    tnr_macro = TN_total / (TN_total + FP_total + 1e-9)

    # Per-class metrics
    num_classes = category_logits.shape[1]
    precision_by_class = {}
    recall_by_class = {}
    f1_by_class = {}
    auc_by_class = {}
    mrr_by_class = {}
    accuracy_by_class = {}

    for c in range(num_classes):
        true_indices = (y_true == c)
        if true_indices.sum() == 0:
            accuracy_by_class[c] = 0.0
        else:
            correct_preds = (y_pred[true_indices] == c).sum()
            accuracy_by_class[c] = correct_preds / true_indices.sum()

    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        y_score_class = y_scores[:, c]

        precision_by_class[c] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        recall_by_class[c] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1_by_class[c] = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        try:
            auc_by_class[c] = roc_auc_score(y_true_bin, y_score_class)
        except Exception:
            auc_by_class[c] = float('nan')

        ranks_c = np.argsort(-y_score_class)
        true_indices = np.where(y_true_bin == 1)[0]
        if len(true_indices) > 0:
            mrr_c = np.mean([
                1 / (np.where(ranks_c == idx)[0][0] + 1) for idx in true_indices
            ])
        else:
            mrr_c = 0.0
        mrr_by_class[c] = mrr_c

    return (
        avg_precision,          # 0
        auc_score,              # 1
        mrr,                    # 2  CATEGORY MRR (Comment 5: relabel in the paper)
        recall_macro,           # 3
        acc,                    # 4
        total_category_loss / num_batches if category_criterion else None,  # 5
        link_extras,            # 6  (was {'edge_loss': ...}; now also carries link scores/metrics)
        accuracy_by_class,      # 7
        f1_macro,               # 8
        fpr_macro,              # 9
        fnr_macro,              # 10
        tpr_macro,              # 11
        tnr_macro,              # 12
        hits_at_1,              # 13 CATEGORY Hits@1 == accuracy (Comment 5)
        hits_at_3,              # 14
        hits_at_5,              # 15
        y_true,                 # 16
        y_scores,               # 17
        precision_by_class,     # 18
        auc_by_class,           # 19
        mrr_by_class,           # 20
        recall_by_class         # 21
        )


def find_best_threshold_per_class(y_true, y_scores, class_labels, metric='f1'):
    thresholds_by_class = {}

    for cls in np.unique(class_labels):
        mask = class_labels == cls
        true = y_true[mask]
        scores = y_scores[mask]

        best_thresh, best_score = 0.5, 0.0
        if len(np.unique(true)) < 2:
            thresholds_by_class[cls] = 0.5
            continue

        for t in np.linspace(0.01, 0.99, 99):
            pred = (scores >= t).astype(int)
            score = f1_score(true, pred, zero_division=0)
            if score > best_score:
                best_thresh = t
                best_score = score
        thresholds_by_class[cls] = best_thresh

    return thresholds_by_class
