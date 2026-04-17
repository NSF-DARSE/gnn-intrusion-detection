import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ids-custom', help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='/workspace/data', help='Root directory of dataset')
    args = parser.parse_args()
    return args

# 1. Setup Output Folder
args = get_args()
output_dir = f"improved_network_windows_{args.dataset}"
os.makedirs(output_dir, exist_ok=True)

# 2. Setup Dataset Path
dataset_root = Path(args.dataset_dir)
if args.dataset == 'ids-unsw-full':
    dataset_root = dataset_root / 'graph_unsw_full_10min'
elif args.dataset == 'ids-custom':
    # Prioritize the specific moduleA folder if it exists
    custom_path = dataset_root / 'graph_10min_moduleA'
    if custom_path.exists():
        dataset_root = custom_path
    else:
        # Fallback logic to match learning.py
        if not (dataset_root / "graphs").exists():
            unsw_path = dataset_root / 'graph_unsw_full_10min'
            if (unsw_path / "graphs").exists():
                dataset_root = unsw_path

graphs_dir = dataset_root / "graphs" / "test"
if not graphs_dir.exists():
    raise ValueError(f"Test graph directory not found: {graphs_dir}")

# Load all .npz files in the test directory
graph_files = sorted(list(graphs_dir.glob("*.npz")))
num_windows = len(graph_files)
print(f"Found {num_windows} test windows in {graphs_dir}")

# ---------------------------------------------------------------------
# GCN Integration: Load Predicted Labels
# ---------------------------------------------------------------------
print(f"Loading GCN predictions for {args.dataset}...")
try:
    gcn_preds = np.load(f'test_predictions_{args.dataset}.npy')
    gcn_labels = np.load(f'test_labels_{args.dataset}.npy')
    print("GCN predictions loaded successfully.")
except Exception as e:
    print(f"Warning: could not load GCN predictions ({e}).")
    gcn_preds = None
    gcn_labels = None

# 3. Process Each Window (.npz file)
# We iterate through the files in the test directory
for w, graph_file in enumerate(graph_files):
    data = np.load(graph_file)
    
    # Load components from .npz
    node_labels = data['node_labels']
    edge_index = data['edge_index']
    
    G = nx.DiGraph()
    
    # Node mapping: Since the GCN combines all windows into one large graph, 
    # the predictions are indexed by the total number of nodes across all windows.
    # We need to find the offset for the current window to align predictions.
    # However, for the visual representation of a single window's correctness,
    # we can derive the status based on the labels and predictions of that window's slice.
    
    # Calculate total node offset for this window
    node_offset = 0
    if gcn_preds is not None:
        # Load the predefined node counts instead of reloading every file in a loop
        try:
            window_counts = np.load(f'test_window_node_counts_{args.dataset}.npy')
            node_offset = np.sum(window_counts[:w])
        except Exception as e:
            print(f"Warning: could not load window counts ({e}), falling back to manual scan.")
            for prev_file in graph_files[:w]:
                node_offset += np.load(prev_file)['node_features'].shape[0]
    else:
        # If no predictions, we don't strictly need the offset for the loop, 
        # but we'll keep the logic for consistency.
        node_offset = 0
    
    num_nodes = node_labels.shape[0]
    
    # Determine node status for this window
    # TP: label=1, pred=1 | TN: label=0, pred=0 | FP: label=0, pred=1 | FN: label=1, pred=0
    node_statuses = []
    for i in range(num_nodes):
        label = node_labels[i]
        pred = gcn_preds[node_offset + i] if gcn_preds is not None else label # fallback to ground truth
        
        if label == 1 and pred == 1: status = "TP"
        elif label == 0 and pred == 0: status = "TN"
        elif label == 0 and pred == 1: status = "FP"
        else: status = "FN"
        node_statuses.append(status)

    # Build the graph
    for i in range(num_nodes):
        G.add_node(i, status=node_statuses[i])
        
    for edge in edge_index:
        G.add_edge(edge[0], edge[1])

    # 4. Advanced Visualization Setup
    plt.figure(figsize=(14, 10))
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    plt.title(f"Dataset: {args.dataset} | Window {w + 1} - Nodes: {num_nodes}, Edges: {num_edges}",
              fontsize=20, fontweight='bold', pad=20)

    pos = nx.spring_layout(G, k=0.8, iterations=75, seed=42)

    color_map = {
        'TP': '#ff3333',  # Red
        'TN': '#33cc33',  # Green
        'FP': '#ffaa00',  # Orange
        'FN': '#888888'   # Gray
    }
    shape_map = {
        'TP': 'o',  # Circle
        'TN': 's',  # Square
        'FP': '^',  # Triangle
        'FN': 'X'   # Cross
    }
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0, edge_color="#777777", arrows=True, arrowsize=12)

    # To draw different shapes, we must loop through the status categories
    for status, shape in shape_map.items():
        nodes_of_status = [n for n in G.nodes() if G.nodes[n]['status'] == status]
        if not nodes_of_status:
            continue
            
        # Get sizes for this specific group
        node_sizes = [min(G.degree[n] * 15 + 40, 600) for n in nodes_of_status]
        
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=nodes_of_status, 
            node_color=color_map[status], 
            node_size=node_sizes, 
            node_shape=shape,
            edgecolors='black', 
            linewidths=1.5
        )

    # Use Line2D to create legend markers that match the node shapes
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='TP (True Positive - Malicious)',
               markerfacecolor='#ff3333', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='TN (True Negative - Benign)',
               markerfacecolor='#33cc33', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', label='FP (False Positive - False Alarm)',
               markerfacecolor='#ffaa00', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='X', color='w', label='FN (False Negative - Missed Attack)',
               markerfacecolor='#888888', markersize=12, markeredgecolor='black'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    plt.axis('off')
    output_path = os.path.join(output_dir, f"Test_{w + 1:02d}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

# ---------------------------------------------------------------------
# 5. Load predictions and visualize statistics
# ---------------------------------------------------------------------
if gcn_preds is not None and gcn_labels is not None:
    if gcn_preds.shape != gcn_labels.shape:
        print("Prediction and label shapes differ, skipping summary plot.")
    else:
        status_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for p, l in zip(gcn_preds, gcn_labels):
            if l == 1 and p == 1:
                status_counts['TP'] += 1
            elif l == 0 and p == 0:
                status_counts['TN'] += 1
            elif l == 0 and p == 1:
                status_counts['FP'] += 1
            elif l == 1 and p == 0:
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
