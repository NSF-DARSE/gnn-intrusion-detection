#!/usr/bin/env python3
"""Module C: render test-set graph visualizations from GCN predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


COLORS = {
    "TN": "#2ca02c",
    "TP": "#d62728",
    "FP": "#ffae34",
    "FN": "#7f7f7f",
}


def load_predictions(path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    predictions: Dict[str, Dict[int, Dict[str, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            graph_file = row["graph_file"]
            node_idx = int(row["node_idx"])
            predictions.setdefault(graph_file, {})[node_idx] = {
                "true_label": int(row["true_label"]),
                "pred_label": int(row["pred_label"]),
                "prob_malicious": float(row["prob_malicious"]),
            }
    return predictions


def outcome(true_label: int, pred_label: int) -> str:
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def render_graph(
    graph_file: Path,
    preds: Dict[int, Dict[str, float]],
    idx_to_node: Dict[int, str],
    output_path: Path,
    layout_seed: int,
) -> Dict[str, int]:
    arrays = np.load(graph_file)
    edge_index = arrays["edge_index"]
    node_features = arrays["node_features"]
    active_nodes = [idx for idx in range(node_features.shape[0]) if float(node_features[idx, 0]) != 0.0]

    graph = nx.DiGraph()
    for node_idx in active_nodes:
        pred = preds[node_idx]
        graph.add_node(
            node_idx,
            label=idx_to_node.get(node_idx, str(node_idx)),
            outcome=outcome(pred["true_label"], pred["pred_label"]),
            prob_malicious=pred["prob_malicious"],
            flow_count=float(node_features[node_idx, 0]),
        )

    for src, dst in edge_index.T:
        src_i = int(src)
        dst_i = int(dst)
        if src_i in graph and dst_i in graph:
            graph.add_edge(src_i, dst_i)

    counts = {key: 0 for key in COLORS}
    for node in graph.nodes:
        counts[graph.nodes[node]["outcome"]] += 1

    plt.figure(figsize=(13, 9))
    title = (
        f"GCN Test Graph | {graph_file.name} | "
        f"TP={counts['TP']} TN={counts['TN']} FP={counts['FP']} FN={counts['FN']}"
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=14)
    pos = nx.spring_layout(graph, seed=layout_seed, k=0.9, iterations=100, weight=None)
    node_colors = [COLORS[graph.nodes[n]["outcome"]] for n in graph.nodes]
    node_sizes = [min(120 + graph.nodes[n]["flow_count"] * 8, 900) for n in graph.nodes]

    nx.draw_networkx_edges(graph, pos, alpha=0.28, width=1.0, edge_color="#555555", arrows=True, arrowsize=10)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#111111",
        linewidths=0.8,
    )
    patches = [
        mpatches.Patch(color=COLORS["TP"], label="TP malicious detected"),
        mpatches.Patch(color=COLORS["TN"], label="TN benign"),
        mpatches.Patch(color=COLORS["FP"], label="FP false alarm"),
        mpatches.Patch(color=COLORS["FN"], label="FN missed attack"),
    ]
    plt.legend(handles=patches, loc="upper right", framealpha=0.92)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Module C test-set visualizations from Module B predictions.")
    parser.add_argument("--graph-dir", type=Path, required=True, help="Module A graph artifact directory.")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="Module B test_predictions.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for test-set PNGs.")
    parser.add_argument("--max-graphs", type=int, default=16)
    parser.add_argument("--layout-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions = load_predictions(args.predictions_csv)
    node_mapping = json.loads((args.graph_dir / "node_mapping.json").read_text(encoding="utf-8"))
    idx_to_node = {int(idx): node for node, idx in node_mapping.items()}

    test_files = sorted((args.graph_dir / "graphs" / "test").glob("window_*.npz"))[: args.max_graphs]
    summary: List[Dict[str, object]] = []
    for i, graph_file in enumerate(test_files, start=1):
        key = str(graph_file)
        if key not in predictions:
            raise ValueError(f"Missing predictions for {key}")
        output_path = args.output_dir / f"test_graph_{i:02d}_{graph_file.stem}.png"
        counts = render_graph(graph_file, predictions[key], idx_to_node, output_path, args.layout_seed)
        summary.append({"graph_file": str(graph_file), "output_png": str(output_path), "counts": counts})
        print(f"Wrote: {output_path}")

    (args.output_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
