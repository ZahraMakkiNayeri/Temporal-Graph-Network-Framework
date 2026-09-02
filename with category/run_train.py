"""BiTA / TGN training — server entry point.

Examples:
    python run_train.py --pilot                          # 1 seed, 5 epochs, timing check
    python run_train.py                                  # full 5-seed suite (resumable)
    python run_train.py --loss ce                        # class-weighted CE variant
    python run_train.py --aggregator tcn --prefix tcn    # ablation aggregators
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import pandas as pd
import torch

from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from utils.evaluation import eval_edge_prediction_with_categories
from utils.losses import FocalLoss
from utils.negative_samplers import (CollisionCheckedRandEdgeSampler,
                                     HistoricalEdgeSampler, InductiveEdgeSampler)
from model.extended_tgn import ExtendedTGN
import matplotlib
matplotlib.use('Agg')  # headless server
import logging as _logging
_logging.getLogger('matplotlib').setLevel(_logging.WARNING)
_logging.getLogger('PIL').setLevel(_logging.WARNING)
import math
import logging
import time
import sys
import torch
import numpy as np
import pickle
import pandas as pd                       # FIX (1c): was missing, pd used below
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# FIX (1c): no Colab-only imports; IN_COLAB is set in the first cell.

# FIX (1f): seeds are now set PER RUN inside the run loop below
# (multi-seed evaluation), instead of a single fixed seed here.

# Dictionary to replace argparse
args = {
    'data': 'new_df',
    'bs': 128,
    'prefix': '',
    'n_degree': 10,
    'n_head': 2,
    'n_epoch': 50,
    'n_layer': 1,
    'lr': 0.0001,
    'patience': 5,
    'n_runs': 5,   # FIX (1f): multi-seed evaluation; report mean ± std
    'drop_out': 0.1,
    'gpu': 0,
    'node_dim': 100,
    'time_dim': 100,
    'backprop_every': 1,
    'use_memory': True,   # FIX (1a): memory MUST be on, otherwise the proposed BiTA aggregator is never instantiated
    'embedding_module': 'graph_attention',
    'message_function': 'mlp',   # FIX (1f): match Eq. 1 of the paper (learnable message function)
    'memory_updater': 'gru',
    'aggregator': 'bigru_transformer',   # FIX (1a): train the proposed BiTA aggregator
    'memory_update_at_end': False,
    'message_dim': 100,
    'memory_dim': 9,
    'different_new_nodes': True,
    'uniform': False,
    'randomize_features': False,
    'use_destination_embedding_in_message': False,
    'use_source_embedding_in_message': False,
    'dyrep': False,
    'num_categories' : 4,
    # FIX (1b): hyperparameters of the learnable aggregators
    'aggregator_hidden_dim': 64,
    'aggregator_n_layers': 1,
    'tcn_kernel_size': 3,
    # MEMORY FIX: bounded per-node message window for the learnable aggregators
    # (bucketed batching handles the rest) — disclosed hyperparameter.
    'aggregator_max_seq_len': 128,
    # R10-5 / R1-17 / R8-minor-8: weight of the category loss in the joint objective
    'lambda_category': 1.0,
    # FIX (Comment 5): candidates per positive for genuine link ranking at test time
    'link_ranking_candidates': 20
}

# ---------------------------------------------------------------------------
# Server CLI: any of the keys below can be overridden from the command line;
# unspecified options keep the paper defaults above.
# ---------------------------------------------------------------------------
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--processed-dir', default='data/processed',
                     help='directory written by preprocessing/preprocess_data.py')
_parser.add_argument('--loss', choices=['focal', 'ce'], default='focal',
                     help='category loss: focal (paper) or class-weighted cross-entropy')
_parser.add_argument('--pilot', action='store_true',
                     help='1 seed, <=5 epochs, 10 ranking candidates — for timing/sanity')
_parser.add_argument('--no-plots', action='store_true')
_parser.add_argument('--balance-train-only', action='store_true',
                     help='balance classes WITHIN the training split, after the '
                          'chronological split (Reviewer 1 #4 / Reviewer 8 #6)')
_parser.add_argument('--streaming-eval', action='store_true',
                     help='additionally evaluate the final model one interaction at '
                          'a time (true online setting, Reviewer 6 #8)')
_parser.add_argument('--streaming-eval-max', type=int, default=2000,
                     help='cap on interactions evaluated in streaming mode (seed 0 only)')
_parser.add_argument('--seq-length-bins', action='store_true', default=True,
                     help='report test metrics per message-sequence-length bin (Experiment 8)')
_parser.add_argument('--no-resume', action='store_true',
                     help='rerun all seeds even if test_metrics_multi_seed.csv has rows')
for _k, _v in list(args.items()):
    if isinstance(_v, bool):
        _parser.add_argument('--' + _k.replace('_', '-'), dest=_k, default=None,
                             type=lambda s: s.lower() in ('1', 'true', 'yes'))
    elif isinstance(_v, (int, float, str)):
        _parser.add_argument('--' + _k.replace('_', '-'), dest=_k, default=None, type=type(_v))
_cli = _parser.parse_args()
for _k in args:
    if getattr(_cli, _k, None) is not None:
        args[_k] = getattr(_cli, _k)
if _cli.pilot:
    args['n_runs'] = 1
    args['n_epoch'] = min(args['n_epoch'], 5)
    args['link_ranking_candidates'] = min(args['link_ranking_candidates'], 10)
    print('PILOT MODE:', {k: args[k] for k in ('n_runs', 'n_epoch', 'link_ranking_candidates')})

# ---------------------------------------------------------------------------
# Load the preprocessed artifacts (server replacement for the notebook state)
# ---------------------------------------------------------------------------
new_df = pd.read_csv(os.path.join(_cli.processed_dir, 'new_df.csv'))
edge_features = np.load(os.path.join(_cli.processed_dir, 'edge_features.npy'))
node_features = np.load(os.path.join(_cli.processed_dir, 'node_features.npy'))
print('Loaded processed data:', new_df.shape, edge_features.shape, node_features.shape)

BATCH_SIZE = args['bs']
NUM_NEIGHBORS = args['n_degree']
NUM_NEG = 1
num_categories = 4
NUM_EPOCH = args['n_epoch']
NUM_HEADS = args['n_head']
DROP_OUT = args['drop_out']
GPU = args['gpu']
DATA = args['data']
NUM_LAYER = args['n_layer']
LEARNING_RATE = args['lr']
NODE_DIM = args['node_dim']
TIME_DIM = args['time_dim']
USE_MEMORY = args['use_memory']
MESSAGE_DIM = args['message_dim']
MEMORY_DIM = args['memory_dim']

# Initialize dictionaries to track category loss and accuracy by type
category_loss_by_type = {i: [] for i in range(num_categories)}
category_accuracy_by_type = {i: [] for i in range(num_categories)}

# FIX (1c): paths no longer hard-coded to Google Drive; relative paths work
# both locally and in Colab. Artifacts land next to the notebook.
saved_models_path = './saved_models/'
saved_checkpoints_path = './saved_checkpoints/'
log_path = './log/'

# Create directories if they don't exist
Path(saved_models_path).mkdir(parents=True, exist_ok=True)
Path(saved_checkpoints_path).mkdir(parents=True, exist_ok=True)
Path(log_path).mkdir(parents=True, exist_ok=True)

# Set up paths for saving models and checkpoints
MODEL_SAVE_PATH = f'{saved_models_path}{args["prefix"]}-{args["data"]}.pth'
# FIX (1f): checkpoints are kept per run so multi-seed runs don't overwrite each other
get_checkpoint_path = lambda epoch: f'{saved_checkpoints_path}{args["prefix"]}-{args["data"]}-run{RUN_IDX}-{epoch}.pth'

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'{log_path}{str(time.time())}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(new_df, edge_features, node_features,
                              different_new_nodes_between_val_and_test=args['different_new_nodes'],
                              randomize_features=args['randomize_features'])

# FIX (Reviewer 6, Comment 4): with BALANCE_CLASSES = False the data keeps its
# original distribution; imbalance is handled here, in the loss, with per-class
# weights computed on the TRAINING split ONLY (no full-week fitting, no
# resampling, no synthetic test distribution). Per-class test support is logged
# so per-class metrics (Table 4) can be read against the real frequencies.
# ----------------------------------------------------------------------
# Reviewer 1 #4 / Reviewer 8 #6: class balancing performed ONLY inside the
# training partition, AFTER the chronological split. Validation and test keep
# their natural distribution and are never resampled, so no duplicated
# interaction can cross a split boundary.
# ----------------------------------------------------------------------
if _cli.balance_train_only:
    import collections as _collections
    _rng_bal = np.random.RandomState(0)
    _counts = _collections.Counter(train_data.labels.astype(int))
    _target = int(np.median(list(_counts.values())))
    _keep = []
    for _c, _n in _counts.items():
        _idx = np.where(train_data.labels.astype(int) == _c)[0]
        _keep.append(_rng_bal.choice(_idx, _target, replace=_n < _target))
    _keep = np.sort(np.concatenate(_keep))          # keep chronological order
    train_data = type(train_data)(
        train_data.sources[_keep], train_data.destinations[_keep],
        train_data.timestamps[_keep], train_data.edge_idxs[_keep],
        train_data.labels[_keep])
    logger.info('Train-only balancing: {} -> {} interactions, target {}/class'.format(
        len(_keep), len(train_data.sources), _target))
    logger.info('  (validation and test splits are untouched)')

_n_cat = args['num_categories']
_train_counts = np.bincount(train_data.labels.astype(int), minlength=_n_cat)
train_class_weights_np = len(train_data.labels) / (_n_cat * np.maximum(_train_counts, 1))
logger.info('Train class counts: {} -> loss weights: {}'.format(
    _train_counts.tolist(), [round(float(w), 3) for w in train_class_weights_np]))
logger.info('Test class support: {}'.format(
    np.bincount(test_data.labels.astype(int), minlength=_n_cat).tolist()))
logger.info('New-node test class support: {}'.format(
    np.bincount(new_node_test_data.labels.astype(int), minlength=_n_cat).tolist()))
#DATA,different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features
# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args['uniform'])

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args['uniform'])

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

all_runs_summary = []   # FIX (1f): collect per-seed test metrics for mean ± std

# Resume support for shared servers / preemptable jobs: seeds whose row is
# already in test_metrics_multi_seed.csv are skipped unless --no-resume.
_completed_runs = set()
if os.path.exists('./test_metrics_multi_seed.csv') and not _cli.no_resume:
    _prev = pd.read_csv('./test_metrics_multi_seed.csv')
    all_runs_summary.extend(_prev.to_dict('records'))
    _completed_runs = set(int(r) for r in _prev['run'])
    print(f'Resuming: seeds {sorted(_completed_runs)} already completed')

for i in range(args['n_runs']):
    results_path = "results/{}_{}.pkl".format(args['prefix'], i) if i > 0 else "results/{}.pkl".format(args['prefix'])
    Path("results/").mkdir(parents=True, exist_ok=True)

    # FIX (1f): one seed per run — the reviewer confirmed the released code
    # evaluated a single run with a fixed seed.
    if i in _completed_runs:
        continue
    RUN_IDX = i
    torch.manual_seed(i)
    np.random.seed(i)
    logger.info('===== Run {} (seed {}) ====='.format(i, i))

    # Initialize Model
    tgn = ExtendedTGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args['memory_update_at_end'],
              embedding_module_type=args['embedding_module'],
              message_function=args['message_function'],
              aggregator_type=args['aggregator'],
              aggregator_hidden_dim=args['aggregator_hidden_dim'],   # FIX (1b)
              aggregator_n_layers=args['aggregator_n_layers'],       # FIX (1b)
              tcn_kernel_size=args['tcn_kernel_size'],               # FIX (1b)
              aggregator_max_seq_len=args['aggregator_max_seq_len'],  # MEMORY FIX
              memory_updater_type=args['memory_updater'],
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args['use_destination_embedding_in_message'],
              use_source_embedding_in_message=args['use_source_embedding_in_message'],
              dyrep=args['dyrep'], num_categories=num_categories)

    criterion = torch.nn.BCELoss()
    if _cli.loss == 'focal':
        category_criterion = FocalLoss(  # FIX (Comment 4): per-class alpha from the TRAIN split
            alpha=torch.tensor(train_class_weights_np, dtype=torch.float32, device=device),
            gamma=2.0)
    else:
        category_criterion = torch.nn.CrossEntropyLoss(  # class-weighted CE variant
            weight=torch.tensor(train_class_weights_np, dtype=torch.float32, device=device))
    # ------------------------------------------------------------------
    # Reviewer 1 #9 / Reviewer 6 #6g / Reviewer 8 #3: exact per-module
    # trainable-parameter breakdown. The previously reported 2,381 was a
    # configuration value, not a parameter count.
    # ------------------------------------------------------------------
    if i == 0:
        _rows, _total = [], 0
        for _name, _mod in tgn.named_children():
            _p = sum(q.numel() for q in _mod.parameters() if q.requires_grad)
            _total += _p
            _rows.append((_name, _p))
        _direct = sum(q.numel() for q in tgn.parameters() if q.requires_grad)
        logger.info('--- trainable parameters per module ---')
        for _name, _p in sorted(_rows, key=lambda r: -r[1]):
            logger.info('  {:32s} {:>10,}'.format(_name, _p))
        logger.info('  {:32s} {:>10,}'.format('TOTAL', _direct))
        pd.DataFrame(_rows + [('TOTAL', _direct)],
                     columns=['module', 'trainable_parameters']).to_csv(
            './parameter_breakdown.csv', index=False)

    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    new_nodes_val_accuracy = []
    new_nodes_val_mrr = []
    val_aps = []
    val_accuracy = []
    val_mrrs = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args['patience'])
    val_link_extras_by_epoch = []   # FIX (1d): validation link scores per epoch
    early_stopped = False
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        for k in range(0, num_batch, args['backprop_every']):
            loss = 0
            optimizer.zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args['backprop_every']):
                batch_idx = k + j

                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                categories_batch = train_data.labels[start_idx:end_idx]

                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                tgn = tgn.train()
                #    def compute_edge_probabilities_and_categories(
                #self, source_nodes, destination_nodes, negative_nodes,
                #pos_edge_times, neg_edge_times, edge_idxs, n_neighbors=20):
                pos_prob, neg_prob, category_logits = tgn.compute_edge_probabilities_and_categories(
                    sources_batch,
                    destinations_batch,
                    negatives_batch,
                    timestamps_batch,
                    edge_idxs_batch,              #
                    n_neighbors=20                #
                )
                category_loss = category_criterion(category_logits, torch.tensor(categories_batch, dtype=torch.long, device=device))
                # FIX: accumulate BOTH losses inside the backprop_every loop
                # (previously category_loss was overwritten and only the last
                # inner batch's category loss was optimized).
                # Joint objective L = L_link + lambda * L_cat (paper Eq. 17).
                loss += (criterion(pos_prob.squeeze(), pos_label)
                         + criterion(neg_prob.squeeze(), neg_label)
                         + args['lambda_category'] * category_loss)

            loss = loss / args['backprop_every']

            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Track category loss by type
            categories_batch_tensor = torch.tensor(categories_batch, dtype=torch.long, device=device)
            for category in range(num_categories):
              mask = (categories_batch == category)
              if mask.any():
                category_loss_by_type[category].append(category_criterion(
                    category_logits[mask], categories_batch_tensor[mask]).item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = tgn.memory.backup_memory()

        (val_ap,
        val_auc,
        val_mrr,
        val_recall,
        val_category_accuracy,
        val_category_loss,
        val_link_extras,
        val_category_accuracy_by_type,
        val_f1,
        val_fpr,
        val_fnr,
        val_tpr,
        val_tnr,
        val_hits_at_1,
        val_hits_at_3,
        val_hits_at_5,
        val_all_labels,
        val_all_scores,
        val_category_precision_by_type,
        val_category_auc_by_type,
        val_category_mrr_by_type,
        val_category_recall_by_type)= eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=val_rand_sampler, data=val_data, n_neighbors=NUM_NEIGHBORS, edge_criterion=criterion, category_criterion= category_criterion)

        #criterion = torch.nn.BCELoss()
        #category_criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # FIX (1d): keep this epoch's validation link scores for threshold tuning
        val_link_extras_by_epoch.append(val_link_extras)

        # Update the tracking dictionaries for validation
        for category in range(num_categories):
            # Check if the key exists in val_link_extras and val_category_accuracy_by_type
            if category in val_link_extras and category in val_category_accuracy_by_type:
                category_loss_by_type[category].append(val_link_extras[category])
                category_accuracy_by_type[category].append(val_category_accuracy_by_type[category])
            else:
                # Optionally add NaN or skip
                category_loss_by_type[category].append(np.nan)
                category_accuracy_by_type[category].append(np.nan)


        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            tgn.memory.restore_memory(train_memory_backup)

        # Validate on unseen nodes
        # FIX (Reviewer 6, Comment 2): this call previously passed val_rand_sampler
        # (old-node negatives) although nn_val_rand_sampler was created for it,
        # inflating the inductive validation scores.
        (nn_val_ap,
        nn_val_auc,
        nn_val_mrr,
        nn_val_recall,
        nn_val_category_accuracy,
        nn_val_category_loss,
        nn_val_link_extras,
        nn_val_category_accuracy_by_type,
        nn_val_f1,
        nn_val_fpr,
        nn_val_fnr,
        nn_val_tpr,
        nn_val_tnr,
        nn_val_hits_at_1,
        nn_val_hits_at_3,
        nn_val_hits_at_5,
        nn_val_all_labels,
        nn_val_all_scores,
        nn_val_category_precision_by_type,
        nn_val_category_auc_by_type,
        nn_val_category_mrr_by_type,
        nn_val_category_recall_by_type) = eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=nn_val_rand_sampler, data=new_node_val_data, n_neighbors=NUM_NEIGHBORS, edge_criterion=criterion, category_criterion= category_criterion)

        if USE_MEMORY:
            # Restore memory we had at the end of validation
            tgn.memory.restore_memory(val_memory_backup)

        new_nodes_val_aps.append(nn_val_ap)
        new_nodes_val_accuracy.append(nn_val_category_accuracy)
        new_nodes_val_mrr.append(nn_val_mrr)
        val_aps.append(val_ap)
        val_accuracy.append(val_category_accuracy)
        val_mrrs.append(val_mrr)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "new_nodes_val_mrr":new_nodes_val_mrr,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        df_r = pd.DataFrame({
            "val_aps": val_aps,
            "new_nodes_val_aps": new_nodes_val_aps,
            "new_nodes_val_mrr":new_nodes_val_mrr,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
        })

        df_r.to_csv('./training_metrics_edgcl.csv', index=True)

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info(
            'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
        logger.info(
            'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
        logger.info(
            'val mrr: {}, new node val mrr: {}'.format(val_mrr, nn_val_mrr))
        logger.info(
            'val category accuracy: {}, new node val category accuracy: {}'.format(val_category_accuracy, nn_val_category_accuracy))
        logger.info(
            'val category loss: {}, new node val category loss: {}'.format(val_category_loss, nn_val_category_loss))

        # Early stopping
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgn.eval()
            early_stopped = True
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    # ------------------------------------------------------------------
    # FIX (Reviewer 6, 1d): DISCLOSED threshold-tuning protocol.
    # The link-classification threshold is tuned (F1-optimal) on the
    # VALIDATION split only, using the validation scores of the epoch
    # whose model is used at test time (best epoch under early stopping,
    # otherwise the final epoch). It is then applied unchanged to the
    # test split. This protocol must be stated in the manuscript
    # wherever threshold-based link metrics are reported.
    # ------------------------------------------------------------------
    _sel_epoch = early_stopper.best_epoch if early_stopped else len(val_link_extras_by_epoch) - 1
    _val_extras = val_link_extras_by_epoch[_sel_epoch]
    _val_scores = np.concatenate([_val_extras['pos_scores'], _val_extras['neg_scores']])
    _val_labels = np.concatenate([np.ones_like(_val_extras['pos_scores']),
                                  np.zeros_like(_val_extras['neg_scores'])])
    best_link_threshold, _best_f1 = 0.5, -1.0
    for _t in np.linspace(0.01, 0.99, 99):
        _f1 = f1_score(_val_labels, (_val_scores >= _t).astype(int), zero_division=0)
        if _f1 > _best_f1:
            best_link_threshold, _best_f1 = _t, _f1
    logger.info('Link threshold tuned on validation (epoch {}): {:.2f} (val F1 = {:.4f})'.format(
        _sel_epoch, best_link_threshold, _best_f1))

    ### Test - np.mean(val_ap), np.mean(val_auc), category_accuracy, predicted_positive_edges
    tgn.embedding_module.neighbor_finder = full_ngh_finder

    (test_ap,
     test_auc,
     test_mrr,
     test_recall,
     category_accuracy_test,
     test_category_loss,
     test_link_extras,
     test_category_accuracy_by_type,
     test_f1,
     test_fpr,
     test_fnr,
     test_tpr,
     test_tnr,
     test_hits_at_1,
     test_hits_at_3,
     test_hits_at_5,
     test_all_labels,
     test_all_scores,
     test_category_precision_by_type,
     test_category_auc_by_type,
     test_category_mrr_by_type,
     test_category_recall_by_type)  = eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=test_rand_sampler, data=test_data, n_neighbors=NUM_NEIGHBORS, edge_criterion=criterion, category_criterion= category_criterion, link_ranking_candidates=args['link_ranking_candidates'],
        # shared candidate pool -> transductive and inductive ranking comparable
        candidate_destinations=np.unique(full_data.destinations),
        seq_length_bins=_cli.seq_length_bins)

    for category in range(num_categories):
      if category in test_link_extras and category in test_category_accuracy_by_type:
        category_loss_by_type[category].append(test_link_extras[category])
        category_accuracy_by_type[category].append(test_category_accuracy_by_type[category])
      else:
        category_loss_by_type[category].append(np.nan)
        category_accuracy_by_type[category].append(np.nan)




    # FIX (1d): threshold-based link metrics on TEST using the validation-tuned
    # threshold (disclosed protocol), alongside the rank-based AUC/AP above.
    _test_scores = np.concatenate([test_link_extras['pos_scores'], test_link_extras['neg_scores']])
    _test_labels = np.concatenate([np.ones_like(test_link_extras['pos_scores']),
                                   np.zeros_like(test_link_extras['neg_scores'])])
    _test_pred = (_test_scores >= best_link_threshold).astype(int)
    test_link_acc = accuracy_score(_test_labels, _test_pred)
    test_link_prec = precision_score(_test_labels, _test_pred, zero_division=0)
    test_link_rec = recall_score(_test_labels, _test_pred, zero_division=0)
    test_link_f1 = f1_score(_test_labels, _test_pred, zero_division=0)
    logger.info('Test LINK metrics @ tuned threshold {:.2f} -- acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(
        best_link_threshold, test_link_acc, test_link_prec, test_link_rec, test_link_f1))

    # FIX (Reviewer 6, Comment 5): genuine link-ranking metrics (the tuple's
    # MRR / Hits@K are CATEGORY metrics and are logged as such above).
    if test_link_extras.get('link_ranking'):
        _lr = test_link_extras['link_ranking']
        logger.info('Test LINK RANKING (1 positive vs {} candidates) -- MRR: {:.4f}, '
                    'Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@10: {:.4f}'.format(
                        _lr['num_candidates'], _lr['mrr'], _lr['hits@1'], _lr['hits@3'], _lr['hits@10']))

    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    # Test on unseen nodes # np.mean(val_ap), np.mean(val_auc), category_accuracy, predicted_positive_edges, category_loss
    (nn_test_ap,
     nn_test_auc,
     nn_test_mrr,
     nn_test_recall,
     category_accuracy_nn_test,
     category_loss_nn_test,
     nn_test_link_extras,
     category_accuracy_by_type_nn_test,
     nn_test_f1,
     nn_test_fpr,
     nn_test_fnr,
     nn_test_tpr,
     nn_test_tnr,
     nn_test_hits_at_1,
     nn_test_hits_at_3,
     nn_test_hits_at_5,
     nn_test_all_labels,
     nn_test_all_scores,
     nn_test_category_precision_by_type,
     nn_test_category_auc_by_type,
     nn_test_category_mrr_by_type,
     nn_test_category_recall_by_type) = eval_edge_prediction_with_categories(model=tgn,
                                                   negative_edge_sampler=nn_test_rand_sampler,
                                                   data=new_node_test_data,
                                                   n_neighbors=NUM_NEIGHBORS,
                                                   edge_criterion=criterion,
                                                   category_criterion= category_criterion,
        link_ranking_candidates=args['link_ranking_candidates'],
        # shared candidate pool -> transductive and inductive ranking comparable
        candidate_destinations=np.unique(full_data.destinations),
        seq_length_bins=_cli.seq_length_bins)

    logger.info(
        'Test statistics: Old nodes -- auc: {}, ap: {}, mrr: {}, recall : {}'.format(test_auc, test_ap, test_mrr, test_recall))
    logger.info(
        'Test statistics: New nodes -- auc: {}, ap: {}, mrr: {}, recall: {}'.format(nn_test_auc, nn_test_ap, nn_test_mrr, nn_test_recall))
    logger.info('Test statistics: Old nodes -- accuracy: {}, loss: {}'.format(category_accuracy_test, test_category_loss))
    logger.info('Test statistics: New nodes -- accuracy: {}, loss: {}'.format(category_accuracy_nn_test, category_loss_nn_test))
    logger.info('Test statistics: Old nodes -- accuracy_types: {}, edge_loss: {}'.format(test_category_accuracy_by_type, test_link_extras['edge_loss']))
    logger.info('Test statistics: New nodes -- accuracy_types: {}, edge_loss: {}'.format(category_accuracy_by_type_nn_test, nn_test_link_extras['edge_loss']))
    logger.info('Test statistics: Old nodes -- f1: {}, fpr: {}, fnr: {}, tpr: {}, tnr: {}'.format(test_f1, test_fpr, test_fnr, test_tpr, test_tnr))
    logger.info('Test statistics: New nodes -- f1: {}, fpr: {}, fnr: {}, tpr: {}, tnr: {}'.format(nn_test_f1, nn_test_fpr, nn_test_fnr, nn_test_tpr, nn_test_tnr))
    logger.info('Test statistics: Old nodes -- hits@1: {}, hits@3: {}, hits@5: {}'.format(test_hits_at_1, test_hits_at_3, test_hits_at_5))
    logger.info('Test statistics: New nodes -- hits@1: {}, hits@3: {}, hits@5: {}'.format(nn_test_hits_at_1, nn_test_hits_at_3, nn_test_hits_at_5))
    logger.info('Test statistics: Old nodes -- category precision by class: {}, category AUC by class: {}, CATEGORY MRR by class: {}, category recall by class: {}'.format(test_category_precision_by_type, test_category_auc_by_type, test_category_mrr_by_type, test_category_recall_by_type))
    logger.info('Test statistics: New nodes -- category precision by class: {}, category AUC by class: {}, CATEGORY MRR by class: {}, category recall by class: {}'.format(nn_test_category_precision_by_type, nn_test_category_auc_by_type, nn_test_category_mrr_by_type, nn_test_category_recall_by_type))
    #logger.info('Test statistics: Old nodes -- all labels: {}, all scores: {}'.format(test_all_labels, test_all_scores))
    #logger.info('Test statistics: New nodes -- all labels: {}, all scores: {}'.format(nn_test_all_labels, nn_test_all_scores))

    # ------------------------------------------------------------------
    # FIX (Reviewer 6, Comment 2): evaluation under harder negative sampling.
    # The protocol above (RandEdgeSampler) draws uniform random negatives,
    # which Poursafaei et al. [15] showed to inflate AUC/AP. The final model
    # is therefore ALSO evaluated on the test split with:
    #   random_cc  — uniform random negatives, collision-checked
    #   historical — negatives from edges observed in train ∪ val
    #   inductive  — negatives from test-period edges never seen in train ∪ val
    # Memory is restored before every pass so all strategies see the
    # identical model/memory state. Report these numbers alongside (not
    # instead of) the random protocol, and state the protocol in the paper.
    # ------------------------------------------------------------------
    # FIX: moved here — nn_test_link_extras exists only after the new-node test eval
    if nn_test_link_extras.get('link_ranking'):
        _lr = nn_test_link_extras['link_ranking']
        logger.info('New-node Test LINK RANKING (1 positive vs {} candidates) -- MRR: {:.4f}, '
                    'Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@10: {:.4f}'.format(
                        _lr['num_candidates'], _lr['mrr'], _lr['hits@1'], _lr['hits@3'], _lr['hits@10']))

    # ------------------------------------------------------------------
    # Artifacts for the manuscript: Experiment 8 sequence-length bins,
    # the category confusion matrix (R1-minor 6) and PR/ROC curve data
    # (R1-minor 5), saved per seed.
    # ------------------------------------------------------------------
    os.makedirs('./artifacts', exist_ok=True)
    for _tag, _ex in (('test', test_link_extras), ('nntest', nn_test_link_extras)):
        if _ex.get('seq_length_bins'):
            pd.DataFrame(_ex['seq_length_bins']).T.rename_axis('bin').to_csv(
                f'./artifacts/seq_length_bins_{_tag}_run{i}.csv')
            logger.info('{} metrics by message-sequence length: {}'.format(
                _tag, {k: round(v.get('auc', float('nan')), 4)
                       for k, v in _ex['seq_length_bins'].items()}))
        if _ex.get('category_confusion_matrix') is not None:
            np.savetxt(f'./artifacts/confusion_matrix_{_tag}_run{i}.csv',
                       _ex['category_confusion_matrix'], fmt='%d', delimiter=',')
        _pr, _roc = _ex.get('link_pr_curve'), _ex.get('link_roc_curve')
        if _pr is not None and _roc is not None:
            np.savez_compressed(f'./artifacts/curves_{_tag}_run{i}.npz',
                                pr_precision=_pr[0], pr_recall=_pr[1],
                                roc_fpr=_roc[0], roc_tpr=_roc[1],
                                category_scores=_ex['category_scores'],
                                category_true=_ex['category_true'])

    # Streaming (online) evaluation — Reviewer 6, Comment 8. One interaction at
    # a time, the setting a deployed SOC would face. Off by default: it is slow.
    if _cli.streaming_eval and i == 0:
        # Batch size 1 means one forward pass per interaction, so this is ~50x
        # slower than batched evaluation. It is a protocol demonstration, not a
        # headline number: run it for the first seed only and over a bounded
        # prefix of the test split (state the prefix length in the paper).
        _k = min(_cli.streaming_eval_max, len(test_data.sources))
        _stream_data = type(test_data)(
            test_data.sources[:_k], test_data.destinations[:_k],
            test_data.timestamps[:_k], test_data.edge_idxs[:_k],
            test_data.labels[:_k])
        logger.info('Streaming evaluation over the first {} test interactions '
                    '(batch size 1)...'.format(_k))
        if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)
        _stream = eval_edge_prediction_with_categories(
            model=tgn, negative_edge_sampler=test_rand_sampler, data=_stream_data,
            n_neighbors=NUM_NEIGHBORS, link_threshold=best_link_threshold,
            streaming=True)
        logger.info('STREAMING (batch size 1) test -- auc: {:.4f}, ap: {:.4f}, '
                    'category acc: {:.4f}, link f1@thr: {:.4f}'.format(
                        _stream[1], _stream[0], _stream[4], _stream[6]['link_f1']))
        if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)

    hist_src = np.concatenate([train_data.sources, val_data.sources])
    hist_dst = np.concatenate([train_data.destinations, val_data.destinations])
    ns_strategies = {
        'random_cc': CollisionCheckedRandEdgeSampler(full_data.sources, full_data.destinations, seed=2),
        'historical': HistoricalEdgeSampler(hist_src, hist_dst, full_data.destinations, seed=2),
        'inductive': InductiveEdgeSampler(hist_src, hist_dst,
                                          test_data.sources, test_data.destinations,
                                          full_data.destinations, seed=2),
    }
    ns_results = {}
    for ns_name, ns_sampler in ns_strategies.items():
        if USE_MEMORY:
            tgn.memory.restore_memory(val_memory_backup)
        _r = eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=ns_sampler,
                                                  data=test_data, n_neighbors=NUM_NEIGHBORS,
                                                  link_threshold=best_link_threshold)
        ns_results[ns_name] = {'auc': _r[1], 'ap': _r[0],
                               'link_acc': _r[6]['link_accuracy'], 'link_f1': _r[6]['link_f1']}
        logger.info('Test LINK metrics under {} negatives -- auc: {:.4f}, ap: {:.4f}, '
                    'acc@thr: {:.4f}, f1@thr: {:.4f}'.format(
                        ns_name, _r[1], _r[0], _r[6]['link_accuracy'], _r[6]['link_f1']))
    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    # Save results for this run
    pickle.dump({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "test_ap": test_ap,
        "test_mrr": test_mrr,
        "new_node_test_ap": nn_test_ap,
        "new_node_test_mrr":nn_test_mrr,
        "category_loss_new_node_test": category_loss_nn_test,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    pickle.dump({
    "category_loss_by_type": category_loss_by_type,
    "category_accuracy_by_type": category_accuracy_by_type,
    }, open('results/category_metrics.pkl', 'wb'))

    #df_category_metrics = pd.DataFrame({
    #f"category_{i}_loss": category_loss_by_type[i] for i in range(num_categories)})

    #df_category_metrics.to_csv('/content/category_metrics.csv', index=True)

    df_test = pd.DataFrame({
        "val_aps": val_aps,
        "new_nodes_val_aps": new_nodes_val_aps,
        "test_ap": test_ap,
        "test_mrr": test_mrr,
        "new_node_test_ap": nn_test_ap,
        "new_node_test_mrr":nn_test_mrr,
        "category_loss_new_node_test": category_loss_nn_test,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times})

    df_test.to_csv('./test_metrics_edgcl.csv', index=True)

    logger.info('Saving TGN model')
    if USE_MEMORY:
        # Restore memory at the end of validation (save a model which is ready for testing)
        tgn.memory.restore_memory(val_memory_backup)
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH.replace('.pth', f'-run{i}.pth'))  # FIX (1f)
    logger.info('TGN model saved')

    # FIX (1f): collect this run's test metrics for cross-seed aggregation
    all_runs_summary.append({
        'run': i, 'seed': i,
        'test_auc': test_auc, 'test_ap': test_ap,
        'test_category_mrr': test_mrr,   # FIX (Comment 5): relabelled — 4-class category MRR
        **({'test_link_mrr': test_link_extras['link_ranking']['mrr'],
            'test_link_hits1': test_link_extras['link_ranking']['hits@1'],
            'test_link_hits3': test_link_extras['link_ranking']['hits@3'],
            'test_link_hits10': test_link_extras['link_ranking']['hits@10']}
           if test_link_extras.get('link_ranking') else {}),
        'test_category_accuracy': category_accuracy_test, 'test_f1_macro': test_f1,
        'test_link_acc': test_link_acc, 'test_link_precision': test_link_prec,
        'test_link_recall': test_link_rec, 'test_link_f1': test_link_f1,
        'tuned_link_threshold': best_link_threshold,
        'nn_test_auc': nn_test_auc, 'nn_test_ap': nn_test_ap,
        'nn_test_category_mrr': nn_test_mrr,   # FIX (Comment 5): relabelled
        **({'nn_test_link_mrr': nn_test_link_extras['link_ranking']['mrr'],
            'nn_test_link_hits1': nn_test_link_extras['link_ranking']['hits@1'],
            'nn_test_link_hits3': nn_test_link_extras['link_ranking']['hits@3'],
            'nn_test_link_hits10': nn_test_link_extras['link_ranking']['hits@10']}
           if nn_test_link_extras.get('link_ranking') else {}),
        # FIX (Comment 2): link metrics under harder negative-sampling protocols
        **{f'test_{m}_{ns_name}': v for ns_name, d in ns_results.items() for m, v in d.items()},
        # Per-class metrics flattened so Table 4 can be reported as mean +/- std
        # across seeds (transductive = test_, inductive = nntest_).
        **{f'test_c{c}_{name}': float(d[c])
           for name, d in (('acc', test_category_accuracy_by_type),
                           ('precision', test_category_precision_by_type),
                           ('auc', test_category_auc_by_type),
                           ('recall', test_category_recall_by_type))
           for c in d},
        **{f'nntest_c{c}_{name}': float(d[c])
           for name, d in (('acc', category_accuracy_by_type_nn_test),
                           ('precision', nn_test_category_precision_by_type),
                           ('auc', nn_test_category_auc_by_type),
                           ('recall', nn_test_category_recall_by_type))
           for c in d},
    })

    # Persist after EVERY seed, not only at the end: a job killed by the wall
    # clock (e.g. a 4-hour partition) then keeps its completed seeds, and
    # resubmitting resumes with the ones still missing.
    pd.DataFrame(all_runs_summary).set_index('run').to_csv('./test_metrics_multi_seed.csv')
    logger.info('Wrote partial summary after seed {} ({} seed(s) complete)'.format(
        i, len(all_runs_summary)))

# ----------------------------------------------------------------------
# FIX (Reviewer 6, 1f): aggregate results across seeds — the manuscript
# should report these mean ± std values, not a single fixed-seed run.
# ----------------------------------------------------------------------
summary_df = pd.DataFrame(all_runs_summary).set_index('run')
print(summary_df)
print('\n=== Mean ± std across {} seeds ==='.format(len(summary_df)))
_stats = summary_df.drop(columns=['seed']).agg(['mean', 'std']).T
for _metric, _row in _stats.iterrows():
    print('{:26s} {:.4f} ± {:.4f}'.format(_metric, _row['mean'], _row['std']))
summary_df.to_csv('./test_metrics_multi_seed.csv')

import matplotlib.pyplot as plt

# Define the fontsize for the numbers
number_fontsize = 25

# Plotting the results
plt.figure(figsize=(15, 10))

# Plot training loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('Loss', fontsize=number_fontsize)
plt.title('Training Loss over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

# Plot validation AP
plt.subplot(2, 2, 2)
plt.plot(val_aps, label='Validation AP')
plt.plot(new_nodes_val_aps, label='New Nodes Validation AP')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('Average Precision', fontsize=number_fontsize)
plt.title('Validation AP over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

# Plot validation Accuracy
plt.subplot(2, 2, 3)
plt.plot(val_accuracy, label='Validation Accuracy')
plt.plot(new_nodes_val_accuracy, label='New Nodes Validation Accuracy')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('Average Accuracy', fontsize=number_fontsize)
plt.title('Validation Accuracy over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

# Plot validation MRR
plt.subplot(2, 2, 4)
plt.plot(val_mrrs, label='Validation MRR')
plt.plot(new_nodes_val_mrr, label='New Nodes Validation MRR')
plt.xlabel('Epoch', fontsize=number_fontsize)
plt.ylabel('MRR', fontsize=number_fontsize)
plt.title('Validation MRR over Epochs')
plt.legend()
plt.xticks(fontsize=number_fontsize)
plt.yticks(fontsize=number_fontsize)

plt.tight_layout()
os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved ./figures/training_curves.png')
