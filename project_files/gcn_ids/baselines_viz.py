import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# 1. Setup Output Folder
output_dir = "improved_network_windows"
os.makedirs(output_dir, exist_ok=True)

# 2. Load and Sort the CSV Data
print("Loading dataset: CTU-IoT-Malware-Capture.labeled.csv...")
df = pd.read_csv('CTU-IoT-Malware-Capture.labeled.csv')
df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
df = df.dropna(subset=['ts']).sort_values(by='ts')

# 3. Create 16 Time Windows
num_windows = 16
print(f"Dividing timeline into {num_windows} test windows...")
df['window_id'] = pd.cut(df['ts'], bins=num_windows, labels=False)

global_node_status = {}

# 4. Process Each Window
for w in range(num_windows):
    window_df = df[df['window_id'] == w]
    if window_df.empty:
        continue

    G = nx.DiGraph()

    for _, row in window_df.iterrows():
        src = str(row['id.orig_h'])
        dst = str(row['id.resp_h'])
        label = str(row['label']).strip()

        # Default to TN (True Negative - Benign) if we haven't seen the IP
        if src not in global_node_status: global_node_status[src] = "TN"
        if dst not in global_node_status: global_node_status[dst] = "TN"

        # Apply Mentor's Logic: Map malicious flows to TP (True Positive - Attack)
        # Note: When Andrew finishes the GCN, I will add FP and FN logic here!
        if 'Malicious' in label:
            global_node_status[src] = "TP"
            global_node_status[dst] = "TP"

        G.add_node(src, status=global_node_status[src])
        G.add_node(dst, status=global_node_status[dst])
        G.add_edge(src, dst)

    # 5. Advanced Visualization Setup (Making it clear and recognizable)
    plt.figure(figsize=(14, 10))  # Increased canvas size for clarity

    # NEW DYNAMIC TITLE FORMAT: "Test No. X - Nodes: Y, Edges: Z"
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    plt.title(f"Test No. {w + 1} - Nodes: {num_nodes}, Edges: {num_edges}",
              fontsize=20, fontweight='bold', pad=20)

    # LAYOUT FIX: 'k' acts as the optimal distance between nodes.
    # Increasing 'k' and 'iterations' spreads the graph out so edges don't overlap as much.
    pos = nx.spring_layout(G, k=0.8, iterations=75, seed=42)

    # Color Mapping for the 4 Dimensions
    color_map = {
        'TP': '#ff3333',  # Red for True Positive
        'TN': '#33cc33',  # Green for True Negative
        'FP': '#ffaa00',  # Orange/Yellow for False Positive
        'FN': '#888888'  # Gray for False Negative
    }
    node_colors = [color_map.get(G.nodes[n]['status'], '#33cc33') for n in G.nodes()]

    # Size Mapping: Make busy nodes larger, but cap them at 600 so they don't cover the edges
    degrees = dict(G.degree())
    node_sizes = [min(v * 15 + 40, 600) for v in degrees.values()]

    # DRAWING FIX: Draw edges and nodes separately for better layering
    # Make edges thinner and translucent (alpha=0.3) so we can see through the clutter
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0, edge_color="#777777", arrows=True, arrowsize=12)

    # Draw nodes with a solid black border (edgecolors='black') to make them pop out from the background
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=1.5)

    # 6. Add the 4-Dimension Legend
    tp_patch = mpatches.Patch(color='#ff3333', label='TP (True Positive - Malicious)')
    tn_patch = mpatches.Patch(color='#33cc33', label='TN (True Negative - Benign)')
    fp_patch = mpatches.Patch(color='#ffaa00', label='FP (False Positive - False Alarm)')
    fn_patch = mpatches.Patch(color='#888888', label='FN (False Negative - Missed Attack)')
    plt.legend(handles=[tp_patch, tn_patch, fp_patch, fn_patch], loc='upper right', fontsize=12, framealpha=0.9)

    # Save the high-resolution image
    plt.axis('off')  # Hide the standard chart box/grid
    output_path = os.path.join(output_dir, f"Test_{w + 1:02d}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved Test {w + 1} (Nodes: {num_nodes}, Edges: {num_edges})")
    plt.close()

print(f"\nSuccess! Improved images have been saved to the '{output_dir}' folder.")

# ---------------------------------------------------------------------
# 7. Load predictions from the GCN training script and visualise overall
#    classification statistics. The training script (`learning.py`) saves two
#    NumPy files – `test_predictions.npy` (model predictions) and
#    `test_labels.npy` (ground‑truth labels). We load them here to compute a
#    simple bar‑chart of TP / TN / FP / FN counts.
# ---------------------------------------------------------------------
try:
    import numpy as np
    preds = np.load('test_predictions.npy')
    labels = np.load('test_labels.npy')
    if preds.shape != labels.shape:
        raise ValueError('Prediction and label shapes differ')

    # Derive status per node
    status_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for p, l in zip(preds, labels):
        if l == 1 and p == 1:
            status_counts['TP'] += 1
        elif l == 0 and p == 0:
            status_counts['TN'] += 1
        elif l == 0 and p == 1:
            status_counts['FP'] += 1
        elif l == 1 and p == 0:
            status_counts['FN'] += 1

    # Simple bar chart visualising the four categories
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = list(status_counts.keys())
    values = list(status_counts.values())
    ax.bar(categories, values, color=['#ff3333', '#33cc33', '#ffaa00', '#888888'])
    ax.set_ylabel('Count')
    ax.set_title('Overall GCN Classification Summary')
    summary_path = os.path.join(output_dir, 'overall_classification_summary.png')
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close(fig)
    print(f"Saved overall classification summary to {summary_path}")
except Exception as e:
    # If the prediction files are missing we simply continue – the visualisation
    # of the network windows still works.
    print(f"Warning: could not load GCN predictions ({e}); skipping summary plot.")
