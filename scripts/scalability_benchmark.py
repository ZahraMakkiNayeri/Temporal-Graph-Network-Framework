"""
Scalability and latency benchmark (Experiment 12 / Section 5.19).

Reviewer 1 #18 asks whether the reported latencies include preprocessing, message
sorting, negative sampling, memory updates, category prediction and batching.
This script times the FULL inference path end to end and says so explicitly, and
labels the daily-throughput figure as an idealised ceiling rather than measured
operational throughput (Reviewer 8 minor #7).

    python scripts/scalability_benchmark.py --processed-dir data/processed \
        --model saved_models/<file>.pth
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_neighbor_finder, RandEdgeSampler
from utils.data_processing import get_data, compute_time_statistics
from model.extended_tgn import ExtendedTGN

p = argparse.ArgumentParser()
p.add_argument('--processed-dir', default='data/processed')
p.add_argument('--model', nargs='*', default=None,
               help='a .pth file, several (shell glob), or a directory; the first is used')
p.add_argument('--batch-sizes', default='32,64,128,256,500')
p.add_argument('--graph-sizes', default='2000,3000,5000,6500')
p.add_argument('--n-neighbors', type=int, default=10)
p.add_argument('--num-categories', type=int, default=4)
p.add_argument('--aggregator', default='bigru_transformer',
               help='time a different aggregator, for the per-method runtime table')
p.add_argument('--repeats', type=int, default=200)
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
p.add_argument('--out-dir', default='artifacts')
a = p.parse_args()

new_df = pd.read_csv(os.path.join(a.processed_dir, 'new_df.csv'))
ef = np.load(os.path.join(a.processed_dir, 'edge_features.npy'))
nf = np.load(os.path.join(a.processed_dir, 'node_features.npy'))
device = torch.device(a.device)
_, _, full_data, train_data, _, test_data, _, _ = get_data(
    new_df, ef, nf, different_new_nodes_between_val_and_test=True)
m_s, s_s, m_d, s_d = compute_time_statistics(full_data.sources, full_data.destinations,
                                             full_data.timestamps)
# Inference timing uses the FULL neighbour finder, as the evaluation path does,
# and every finder must span the full node range: a train-only finder has a
# smaller maximum node id and raises IndexError on later nodes.
_max_node = int(max(full_data.sources.max(), full_data.destinations.max()))
_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=_max_node)
model = ExtendedTGN(neighbor_finder=_finder,
    node_features=nf, edge_features=ef, device=device, n_layers=1, n_heads=2,
    dropout=0.1, use_memory=True, message_dimension=100, memory_dimension=nf.shape[1],
    embedding_module_type='graph_attention', message_function='mlp',
    mean_time_shift_src=m_s, std_time_shift_src=s_s, mean_time_shift_dst=m_d,
    std_time_shift_dst=s_d, n_neighbors=a.n_neighbors,
    aggregator_type=a.aggregator, memory_updater_type='gru',
    num_categories=a.num_categories, aggregator_hidden_dim=64,
    aggregator_n_layers=1, aggregator_max_seq_len=128).to(device)
_ckpt = None
if a.model:
    cands = []
    for m in a.model:
        if os.path.isdir(m):
            cands += sorted(glob.glob(os.path.join(m, '*.pth')))
        elif os.path.exists(m):
            cands.append(m)
    if cands:
        try:
            model.load_state_dict(torch.load(cands[0], map_location=device))
            _ckpt = cands[0]
            print('loaded checkpoint:', _ckpt)
        except RuntimeError as e:
            print('WARNING: checkpoint does not match this --processed-dir '
                  '(different edge-feature width). Timing an untrained model instead;\n'
                  '         latency does not depend on the weights, but point --model at the\n'
                  '         saved_models/ of the run that used THIS processed-dir to be exact.')
            print('        ', str(e).splitlines()[0])
if _ckpt is None:
    print('no checkpoint given: timing an untrained model (timings are unaffected)')
model.eval()
print('device:', device, '|', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')
print('Timed path INCLUDES: negative sampling, message construction and sorting,')
print('BiTA aggregation, memory update, embedding, link head and category head.\n')

sampler = RandEdgeSampler(full_data.sources, full_data.destinations)

def timed(bs, n_edges=None):
    src, dst = test_data.sources, test_data.destinations
    ts, ei = test_data.timestamps, test_data.edge_idxs
    if n_edges:
        src, dst, ts, ei = src[:n_edges], dst[:n_edges], ts[:n_edges], ei[:n_edges]
    model.memory.__init_memory__()
    n = min(len(src), a.repeats * bs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in range((n + bs - 1) // bs):
            sl = slice(b * bs, min(n, (b + 1) * bs))
            if sl.stop <= sl.start:
                continue
            _, neg = sampler.sample(sl.stop - sl.start)
            model.compute_edge_probabilities_and_categories(
                src[sl], dst[sl], neg, ts[sl], ei[sl], n_neighbors=a.n_neighbors)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    el = time.perf_counter() - t0
    return el / n * 1000.0, n / el          # ms per edge, edges per second

rows = []
print(f'{"batch":>7} {"ms/edge":>9} {"edges/s":>10}')
for bs in [int(x) for x in a.batch_sizes.split(',')]:
    ms, eps = timed(bs)
    rows.append({'kind': 'batch_size', 'value': bs, 'ms_per_edge': ms, 'edges_per_s': eps})
    print(f'{bs:7d} {ms:9.3f} {eps:10.0f}')

print(f'\n{"edges":>7} {"ms/edge":>9} {"edges/s":>10}')
for ge in [int(x) for x in a.graph_sizes.split(',')]:
    ms, eps = timed(128, n_edges=ge)
    rows.append({'kind': 'graph_size', 'value': ge, 'ms_per_edge': ms, 'edges_per_s': eps})
    print(f'{ge:7d} {ms:9.3f} {eps:10.0f}')

best = max(r['edges_per_s'] for r in rows)
os.makedirs(a.out_dir, exist_ok=True)
_dev = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
_par = sum(q.numel() for q in model.parameters() if q.requires_grad)
for r in rows:
    r['aggregator'] = a.aggregator
    r['device'] = _dev            # timings are only comparable within one device
    r['trainable_parameters'] = _par
pd.DataFrame(rows).to_csv(os.path.join(a.out_dir,
    f'scalability_{a.aggregator}.csv'), index=False)
print(f'\nPeak throughput {best:,.0f} edges/s.')
print(f'Idealised ceiling if sustained continuously: {best*86400/1e6:,.0f} M edges/day')
print('(a linear extrapolation of peak throughput under this hardware and graph')
print(' size, NOT a measured operational rate — label it as such in the paper).')
print('Wrote', os.path.join(a.out_dir, f'scalability_{a.aggregator}.csv'))
print(f'trainable parameters: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}')
