# Module B Documentation

## Purpose

Module B trains, validates, and tests the graph convolutional network (GCN)
using the graph artifacts produced by Module A.

Main implementation file:

- `gcn_ids/learning.py`

## Input

Module B expects a Module A output directory containing:

```text
graphs/train/*.npz
graphs/val/*.npz
graphs/test/*.npz
schema.json
node_mapping.json
manifest.json
```

Each graph file contains:

- `node_features`
- `node_labels`
- `edge_index`
- `edge_features`

## What Module B Does

Module B:

1. loads the train, validation, and test graph files
2. builds a normalized adjacency matrix from `edge_index`
3. trains a 2-layer GCN
4. evaluates the best model on validation and test data
5. writes metrics, predictions, and plots

## Model Summary

The current implementation uses:

- a 2-layer GCN
- hidden dimension: `32`
- dropout: `0.25`
- learning rate: `0.01`
- weight decay: `0.0005`
- default epochs: `50`
- seed: `42`

Loss:

- weighted cross-entropy based on train-set class distribution

## Outputs

Module B writes a results directory like:

```text
module_b_results/
  gcn_model.pt
  metrics.json
  test_predictions.csv
  training_history.csv
  training_curves.png
  confusion_matrix.png
```

### Output meaning

- `gcn_model.pt`
  - saved model weights
- `metrics.json`
  - final metrics and settings
- `test_predictions.csv`
  - per-node test predictions
- `training_history.csv`
  - epoch-by-epoch training and validation history
- `training_curves.png`
  - training/validation loss and accuracy
- `confusion_matrix.png`
  - test confusion matrix

## Run Command

Example:

```bash
python3 gcn_ids/learning.py \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --output-dir data/graph_unsw_full_10min_stratified_clean/module_b_results \
  --epochs 50 \
  --seed 42
```

## Current Result In This Workspace

For the latest stratified-clean UNSW build:

- accuracy: `0.9993`
- malicious precision: `0.9958`
- malicious recall: `1.0000`
- malicious F1: `0.9979`

Confusion matrix:

```text
[[1231, 1],
 [   0, 238]]
```

## Limitations

- Module B currently has less direct automated test coverage than Module A
- evaluation assumes Module A artifacts are valid and correctly structured
- this implementation is intentionally simple and CPU-friendly compared with
  more complex PyTorch Geometric training pipelines
