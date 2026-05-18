# Module C Documentation

## Purpose

Module C generates network graph visualizations from the **test-set**
predictions produced by Module B.

Main implementation file:

- `gcn_ids/graph_viz.py`

## Input

Module C expects:

1. a Module A graph directory
2. a Module B `test_predictions.csv`

Example inputs:

- `data/graph_unsw_full_10min_stratified_clean/`
- `data/graph_unsw_full_10min_stratified_clean/module_b_results/test_predictions.csv`

## What Module C Does

Module C:

1. loads node-level predictions from Module B
2. loads test graph windows from Module A
3. rebuilds each graph structure from `edge_index`
4. colors nodes by outcome:
   - `TP`
   - `TN`
   - `FP`
   - `FN`
5. saves a PNG per graph and a summary JSON file

## Output

Module C writes:

```text
module_c_test_viz/
  test_graph_01_*.png
  test_graph_02_*.png
  ...
  viz_summary.json
```

### Output meaning

- `test_graph_*.png`
  - rendered network graphs for test windows
- `viz_summary.json`
  - mapping between graph files, PNGs, and TP/TN/FP/FN counts

## Color Meaning

The current visualization uses:

- green = `TN`
- red = `TP`
- yellow/orange = `FP`
- gray = `FN`

## Run Command

Example:

```bash
python3 gcn_ids/graph_viz.py \
  --graph-dir data/graph_unsw_full_10min_stratified_clean \
  --predictions-csv data/graph_unsw_full_10min_stratified_clean/module_b_results/test_predictions.csv \
  --output-dir data/graph_unsw_full_10min_stratified_clean/module_c_test_viz \
  --max-graphs 16 \
  --layout-seed 42
```

## Notes

- Module C is designed for **test-set visualization only**
- it depends on valid prediction rows from Module B
- it is best used as an interpretability / communication layer, not as the main
  source of evaluation metrics

## Limitation

The current visualizations are useful for qualitative interpretation, but they
do not replace quantitative evaluation from Module B.
