import argparse
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
import multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ids-custom', help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/workspace/data', help='Root directory of dataset')
    args = parser.parse_args()
    return args


def draw_single_window(win_id, edges, win_preds, win_labels, all_nodes, master_pos, output_dir):
    """Draws a single window using a single CPU core."""
    G_win = nx.Graph()
    G_win.add_nodes_from(all_nodes)
    G_win.add_edges_from(edges)

    node_categories = {}
    for idx, node_id in enumerate(all_nodes):
        if idx < len(win_preds):
            p, label = win_preds[idx], win_labels[idx]
            if label == 1 and p == 1:
                cat = 'TP'
            elif label == 0 and p == 0:
                cat = 'TN'
            elif label == 0 and p == 1:
                cat = 'FP'
            elif label == 1 and p == 0:
                cat = 'FN'
            else:
                cat = 'TN'
        else:
            cat = 'TN'
        node_categories[node_id] = cat

    COLOR_MAP = {'TP': '#d62728', 'TN': '#1f77b4', 'FP': '#ff7f0e', 'FN': '#2ca02c'}
    SHAPE_MAP = {'TP': 'o', 'TN': 's', 'FP': '^', 'FN': 'D'}

    plt.figure(figsize=(14, 11), facecolor='white')

    nx.draw_networkx_edges(G_win, master_pos, alpha=0.4, edge_color='#a3a3a3', width=1.0)

    for category in ['TP', 'TN', 'FP', 'FN']:
        nodelist = [n for n in G_win.nodes() if node_categories.get(n, 'TN') == category]
        if nodelist:
            nx.draw_networkx_nodes(
                G_win, master_pos,
                nodelist=nodelist,
                node_color=COLOR_MAP[category],
                node_shape=SHAPE_MAP[category],
                node_size=800,
                edgecolors='black',
                linewidths=1.5,
            )

    nx.draw_networkx_labels(G_win, master_pos, font_size=11, font_weight='bold', font_color='white')

    active_nodes = len(set([u for u, v in edges]).union(set([v for u, v in edges])))
    title_text = (
        f"Network Intrusion Analysis - Window {win_id}\n"
        f"Total Nodes: {len(all_nodes)} | Active Nodes: {active_nodes} | Connections: {len(edges)}"
    )
    plt.title(title_text, fontsize=16, fontweight='bold', pad=20)

    legend_elements = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor='#d62728', markersize=12, markeredgecolor='black', label='TP (Correct Attack)'),
        mlines.Line2D([], [], color='w', marker='s', markerfacecolor='#1f77b4', markersize=12, markeredgecolor='black', label='TN (Correct Safe)'),
        mlines.Line2D([], [], color='w', marker='^', markerfacecolor='#ff7f0e', markersize=12, markeredgecolor='black', label='FP (False Alarm)'),
        mlines.Line2D([], [], color='w', marker='D', markerfacecolor='#2ca02c', markersize=12, markeredgecolor='black', label='FN (Missed Attack)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9, title="Prediction Profile")

    plt.axis('off')

    output_path = os.path.join(output_dir, f"Window_{win_id:04d}.png")
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return f"Window {win_id} saved."


def main():
    args = get_args()

    output_dir = f"improved_network_windows_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    dataset_root = Path(args.dataset_dir)
    if args.dataset == 'ids-unsw-full':
        dataset_root = dataset_root / 'graph_unsw_full_10min'
    elif args.dataset == 'ids-custom':
        custom_path = dataset_root / 'graph_10min_moduleA'
        if custom_path.exists():
            dataset_root = custom_path
        else:
            if not (dataset_root / "graphs").exists():
                unsw_path = dataset_root / 'graph_unsw_full_10min'
                if (unsw_path / "graphs").exists():
                    dataset_root = unsw_path

    graphs_dir = dataset_root / "graphs" / "test"
    if not graphs_dir.exists():
        raise ValueError(f"Test graph directory not found: {graphs_dir}")

    npz_files = sorted(list(graphs_dir.glob("*.npz")))
    num_windows = len(npz_files)
    print(f"Found {num_windows} test windows in {graphs_dir}")

    preds_dir = Path(f"predictions_{args.dataset}")
    print(f"Loading per-window GCN predictions from {preds_dir}...")
    gcn_has_predictions = preds_dir.exists() and len(list(preds_dir.glob("*_preds.npy"))) > 0
    if not gcn_has_predictions:
        print(f"Error: no per-window predictions found in {preds_dir}.")
        print("Run learning.py first to generate predictions.")
        return

    print("Pass 1: Extracting global network topology...")
    global_G = nx.Graph()
    window_data_cache = []

    for w, npz_file in enumerate(npz_files):
        data = np.load(npz_file)

        if 'edge_index' in data:
            src, dst = data['edge_index']
            edges = list(zip(src.tolist(), dst.tolist()))
        else:
            adj = data['adj_matrix'].item() if data['adj_matrix'].ndim == 0 else data['adj_matrix']
            coo = adj.tocoo()
            edges = list(zip(coo.row.tolist(), coo.col.tolist()))

        num_nodes = data['node_features'].shape[0]

        win_pred_path = preds_dir / f"window_{w:05d}_preds.npy"
        win_label_path = preds_dir / f"window_{w:05d}_labels.npy"
        win_preds = np.load(win_pred_path) if win_pred_path.exists() else np.zeros(num_nodes)
        win_labels = np.load(win_label_path) if win_label_path.exists() else np.zeros(num_nodes)

        global_G.add_edges_from(edges)

        window_data_cache.append((w, edges, win_preds, win_labels))

    all_nodes = list(global_G.nodes())

    print("Calculating Stable Multi-Layer Layout...")
    node_degrees = dict(global_G.degree())
    sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

    core_hubs = [n for n, deg in sorted_nodes[:3]]
    inner_ring = [n for n, deg in sorted_nodes[3:15]] if len(sorted_nodes) > 3 else []
    outer_ring = [n for n, deg in sorted_nodes[15:]] if len(sorted_nodes) > 15 else []

    nlist = [r for r in [core_hubs, inner_ring, outer_ring] if r]
    master_pos = nx.shell_layout(global_G, nlist=nlist)

    print(f"Pass 2: Igniting {mp.cpu_count()} cores for High-Speed Generation...")

    tasks = [
        (w, edges, win_preds, win_labels, all_nodes, master_pos, output_dir)
        for (w, edges, win_preds, win_labels) in window_data_cache
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(draw_single_window, tasks)

    print(f"\nSUCCESS! {len(results)} Pro-Level images saved to '{output_dir}'.")

    status_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for w in range(num_windows):
        pred_path = preds_dir / f"window_{w:05d}_preds.npy"
        label_path = preds_dir / f"window_{w:05d}_labels.npy"
        if not pred_path.exists() or not label_path.exists():
            continue
        win_preds = np.load(pred_path)
        win_labels = np.load(label_path)
        for p, label in zip(win_preds, win_labels):
            if label == 1 and p == 1:
                status_counts['TP'] += 1
            elif label == 0 and p == 0:
                status_counts['TN'] += 1
            elif label == 0 and p == 1:
                status_counts['FP'] += 1
            elif label == 1 and p == 0:
                status_counts['FN'] += 1

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = list(status_counts.keys())
    values = list(status_counts.values())
    ax.bar(categories, values, color=['#ff3333', '#33cc33', '#ffaa00', '#888888'])
    ax.set_ylabel('Count')
    ax.set_title(f'Overall GCN Summary: {args.dataset}')
    summary_path = os.path.join(output_dir, 'overall_classification_summary.png')
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close(fig)
    print(f"Saved overall classification summary to {summary_path}")


if __name__ == '__main__':
    main()
