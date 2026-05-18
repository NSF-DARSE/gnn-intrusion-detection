# GNN Intrusion Detection

This repository contains a small end-to-end graph neural network workflow for
intrusion detection.

The project is organized into three modules:

- **Module A**: clean raw network-flow CSV files and convert them into graph
  datasets
- **Module B**: train and evaluate a graph convolutional network (GCN) on those
  graph datasets
- **Module C**: visualize test-set predictions as network graphs

The goal of this README is to make the repo easy to understand for the next
person who picks it up.

## 1. What The Project Does

The pipeline starts with raw flow tables and ends with model metrics and test
graph visualizations:

```text
Raw CSV flows
  -> Module A (clean + build graph windows)
  -> Module B (train / validate / test GCN)
  -> Module C (visualize test predictions)
```

Each graph window is a directed communication graph:

- **nodes** = IP addresses
- **edges** = observed flows between IPs in a fixed time window

## 2. Main Files You Should Know

### Core code

- `gcn_ids/data_graph.py`
  - Module A
  - cleans input data
  - builds graph windows
  - creates node and edge features
  - writes train / val / test graph artifacts

- `gcn_ids/learning.py`
  - Module B
  - loads graph artifacts from Module A
  - trains a 2-layer GCN
  - writes metrics, predictions, and plots

- `gcn_ids/graph_viz.py`
  - Module C
  - reads Module B predictions
  - creates test-set graph visualizations

- `main.py`
  - convenience entry point that runs Module A -> Module B -> Module C

### Tests

- `tests/test_data_graph.py`
  - current automated tests are strongest for Module A

### Documentation

- `docs/project_onboarding.md`
  - best starting point for a future teammate
- `docs/module_a_documentation.md`
- `docs/module_b_documentation.md`
- `docs/module_c_documentation.md`
- `docs/integration_report.md`

## 3. Repository Structure

```text
gcn_ids/
  __init__.py
  data_graph.py
  learning.py
  graph_viz.py

docs/
  project_onboarding.md
  module_a_documentation.md
  module_b_documentation.md
  module_c_documentation.md
  integration_report.md
  figures/

scripts/
  run_module_a_windows.ps1
  run_module_a_mac.sh
  build_merged_dataset.py
  build_node_window_dataset.py
  visualize_flow_windows.py

tests/
  test_data_graph.py

main.py
requirements.txt
pyproject.toml
README.md
CHANGELOG.md
RELEASE_NOTES.md
LICENSE
```

## 4. Datasets Used In This Project

This repo was used with two datasets during the project:

### A. UNSW-NB15

- mentor-selected main dataset
- used for the main graph/GCN workflow
- latest cleaned graph build:
  - `data/graph_unsw_full_10min_stratified_clean/`

### B. IoT / ids-custom style dataset

- used as a secondary dataset during experimentation
- latest graph build:
  - `data/graph_10min_moduleA_stratified/`

If you are continuing this project, start with the **UNSW-NB15** workflow
unless your team explicitly wants the IoT branch.

## 5. What Module A Produces

Module A writes a graph dataset folder that looks like this:

```text
graph_build/
  graphs/
    train/
      window_*.npz
    val/
      window_*.npz
    test/
      window_*.npz
  manifest.json
  schema.json
  node_mapping.json
  scaler_node.pkl
  scaler_edge.pkl
```

Each `.npz` graph file contains:

- `node_features`
- `node_labels`
- `edge_index`
- `edge_features`

Module B reads these files directly.

## 6. What Module B Produces

Module B writes a results folder like this:

```text
module_b_results/
  gcn_model.pt
  metrics.json
  test_predictions.csv
  training_history.csv
  training_curves.png
  confusion_matrix.png
```

This gives you:

- saved model weights
- test metrics
- node-level predictions
- training/validation curves
- confusion matrix

## 7. What Module C Produces

Module C writes:

```text
module_c_test_viz/
  test_graph_01_*.png
  test_graph_02_*.png
  ...
  viz_summary.json
```

These are **test-set-only** visualizations based on Module B predictions.

## 8. Quick Start

### Install dependencies

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

### Run the full UNSW workflow

```bash
python3 main.py \
  --dataset-dir data/raw/unsw_complete \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --results-dir data/graph_unsw_full_10min_stratified_clean/module_b_results \
  --viz-dir data/graph_unsw_full_10min_stratified_clean/module_c_test_viz \
  --epochs 50 \
  --seed 42
```

## 9. Run Each Module Separately

### Module A only

```bash
python3 gcn_ids/data_graph.py \
  --input-files data/raw/unsw_complete \
  --input-glob "UNSW-NB15_*.csv" \
  --output-dir data/graph_unsw_full_10min_stratified_clean \
  --window 10min \
  --split-ratio 0.6,0.2,0.2 \
  --split-strategy stratified_attack_presence \
  --node-label-rule any_malicious_wins \
  --seed 42
```

### Module B only

```bash
python3 gcn_ids/learning.py \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --output-dir data/graph_unsw_full_10min_stratified_clean/module_b_results \
  --epochs 50 \
  --seed 42
```

### Module C only

```bash
python3 gcn_ids/graph_viz.py \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --predictions-csv data/graph_unsw_full_10min_stratified_clean/module_b_results/test_predictions.csv \
  --output-dir data/graph_unsw_full_10min_stratified_clean/module_c_test_viz \
  --max-graphs 16 \
  --layout-seed 42
```

## 10. Testing

Current automated tests focus on Module A.

Run them with:

```bash
python3 -m pytest -q tests/test_data_graph.py
```

Expected result in this workspace:

```text
8 passed
```

Why these tests matter:

- they verify split logic
- they verify artifact creation
- they verify UNSW raw shard support
- they help confirm that Module B receives the graph inputs it expects

## 11. Current Results In This Workspace

### UNSW stratified-clean run

The current workspace includes a completed end-to-end run under:

- `data/graph_unsw_full_10min_stratified_clean/`

Recent test metrics from Module B:

- accuracy: `0.9993`
- malicious precision: `0.9958`
- malicious recall: `1.0000`
- malicious F1: `0.9979`

Confusion matrix:

```text
[[1231, 1],
 [   0, 238]]
```

### Included figures

- `docs/figures/training_curves.png`
- `docs/figures/confusion_matrix.png`
- `docs/figures/test_graph_example.png`

## 12. Reproducibility Notes

- major commands support a fixed `--seed`
- graph scaling is fit on **train split only**
- metadata is written to `manifest.json`
- model metrics are written to `metrics.json`
- plots are saved as PNGs

## 13. Known Limitations

- automated tests currently cover Module A more thoroughly than Module B or C
- this repo contains traces of earlier project phases and older local artifacts
- the presentation used by the team may summarize the project differently than
  this repository layout; this README describes the code as it exists here

## 14. If You Are Continuing This Project

Start here:

1. read `docs/project_onboarding.md`
2. read this `README.md`
3. inspect `gcn_ids/data_graph.py`, `gcn_ids/learning.py`, and
   `gcn_ids/graph_viz.py`
4. run the Module A tests
5. run the pipeline on the existing UNSW graph build before changing code

## 15. Additional Docs

- `docs/project_onboarding.md`
- `docs/module_a_documentation.md`
- `docs/module_b_documentation.md`
- `docs/module_c_documentation.md`
- `docs/integration_report.md`
- `CHANGELOG.md`
- `RELEASE_NOTES.md`
