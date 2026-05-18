#!/usr/bin/env python3
"""Module B: train, validate, and test a windowed GCN node classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass
class GraphData:
    file: Path
    x: torch.Tensor
    y: torch.Tensor
    adj: torch.Tensor
    edge_index: np.ndarray


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(adj, x)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.matmul(adj, h)
        return self.fc2(h)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def normalized_adjacency(edge_index: np.ndarray, num_nodes: int) -> torch.Tensor:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if edge_index.size:
        for src, dst in edge_index.T:
            adj[int(src), int(dst)] = 1.0
            adj[int(dst), int(src)] = 1.0
    adj += np.eye(num_nodes, dtype=np.float32)
    degree = adj.sum(axis=1)
    inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    inv_sqrt[degree == 0] = 0.0
    adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    return torch.from_numpy(adj)


def load_split(graph_dir: Path, split: str) -> List[GraphData]:
    graphs: List[GraphData] = []
    for file_path in sorted((graph_dir / "graphs" / split).glob("window_*.npz")):
        arrays = np.load(file_path)
        x_np = arrays["node_features"].astype(np.float32)
        y_np = arrays["node_labels"].astype(np.int64)
        edge_index = arrays["edge_index"].astype(np.int64)
        graphs.append(
            GraphData(
                file=file_path,
                x=torch.from_numpy(x_np),
                y=torch.from_numpy(y_np),
                adj=normalized_adjacency(edge_index, x_np.shape[0]),
                edge_index=edge_index,
            )
        )
    if not graphs:
        raise ValueError(f"No graph files found for split '{split}' in {graph_dir}")
    return graphs


def iter_labels(graphs: Iterable[GraphData]) -> np.ndarray:
    return np.concatenate([g.y.numpy() for g in graphs])


def class_weights(train_graphs: List[GraphData]) -> torch.Tensor:
    y = iter_labels(train_graphs)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model: GCN, graphs: List[GraphData], optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    for graph in graphs:
        optimizer.zero_grad()
        logits = model(graph.x, graph.adj)
        loss = criterion(logits, graph.y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        y_true.extend(graph.y.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().numpy().tolist())

    return {"loss": total_loss / len(graphs), "accuracy": accuracy_score(y_true, y_pred)}


@torch.no_grad()
def evaluate(model: GCN, graphs: List[GraphData], criterion: nn.Module) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    rows: List[Dict[str, object]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for graph in graphs:
        logits = model(graph.x, graph.adj)
        loss = criterion(logits, graph.y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item())
        true_np = graph.y.numpy()
        pred_np = preds.numpy()
        prob_np = probs.numpy()
        y_true.extend(true_np.tolist())
        y_pred.extend(pred_np.tolist())

        for node_idx, (label, pred, prob) in enumerate(zip(true_np, pred_np, prob_np)):
            rows.append(
                {
                    "graph_file": str(graph.file),
                    "node_idx": node_idx,
                    "true_label": int(label),
                    "pred_label": int(pred),
                    "prob_malicious": float(prob),
                }
            )

    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "loss": total_loss / len(graphs),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_benign": float(precision[0]),
        "precision_malicious": float(precision[1]),
        "recall_benign": float(recall[0]),
        "recall_malicious": float(recall[1]),
        "f1_benign": float(f1[0]),
        "f1_malicious": float(f1[1]),
        "confusion_matrix": cm.astype(int).tolist(),
        "predictions": rows,
    }


def plot_curves(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Validation")
    axes[0].set_title("Loss Across 50 Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_accuracy"] for h in history], label="Train")
    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="Validation")
    axes[1].set_title("Accuracy Across 50 Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(cm: List[List[int]], output_path: Path) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(arr, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Benign", "Malicious"])
    ax.set_yticks([0, 1], labels=["Benign", "Malicious"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(arr[row, col]), ha="center", va="center", color="black", fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_history(history: List[Dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_predictions(rows: List[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_file", "node_idx", "true_label", "pred_label", "prob_malicious"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a two-layer GCN on Module A graph artifacts.")
    parser.add_argument("--graph-dir", type=Path, required=True, help="Module A output directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for Module B outputs.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_graphs = load_split(args.graph_dir, "train")
    val_graphs = load_split(args.graph_dir, "val")
    test_graphs = load_split(args.graph_dir, "test")

    in_dim = train_graphs[0].x.shape[1]
    model = GCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=2, dropout=args.dropout)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_graphs))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_graphs, optimizer, criterion)
        val_metrics = evaluate(model, val_graphs, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )
        if float(val_metrics["accuracy"]) > best_val:
            best_val = float(val_metrics["accuracy"])
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_graphs, criterion)
    save_history(history, args.output_dir / "training_history.csv")
    save_predictions(test_metrics["predictions"], args.output_dir / "test_predictions.csv")
    plot_curves(history, args.output_dir / "training_curves.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], args.output_dir / "confusion_matrix.png")
    torch.save(model.state_dict(), args.output_dir / "gcn_model.pt")

    metrics = {
        "settings": {
            "graph_dir": str(args.graph_dir),
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "best_val_accuracy": best_val,
        "final_epoch": history[-1],
        "test": {k: v for k, v in test_metrics.items() if k != "predictions"},
        "outputs": {
            "history_csv": str(args.output_dir / "training_history.csv"),
            "test_predictions_csv": str(args.output_dir / "test_predictions.csv"),
            "training_curves_png": str(args.output_dir / "training_curves.png"),
            "confusion_matrix_png": str(args.output_dir / "confusion_matrix.png"),
            "model": str(args.output_dir / "gcn_model.pt"),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
