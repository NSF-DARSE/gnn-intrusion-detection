# Release Notes

## Version 0.1.0

This release packages the course project into a clearer end-to-end GNN intrusion-detection workflow.

### What is included

- raw flow ingestion and graph construction for IDS datasets
- support for UNSW-NB15 original shard files
- support for IoT-style graph dataset generation
- graph-window train/validation/test splitting
- optional stratified splitting for class-imbalanced graph windows
- GCN training, validation, and test evaluation
- training and validation curves
- confusion matrix generation
- test-set graph visualizations
- documentation for Module A and the integrated workflow

### End-user summary

Users can now:

- build graph datasets from supported CSV inputs
- train a GCN from saved graph artifacts
- inspect model results through plots and graph visualizations

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Upgrade / Migration Notes

- users previously relying on chronological-only graph splits can now use `--split-strategy stratified_attack_presence`
- users should use `gcn_ids/graph_viz.py` for Module C visualization in the current integrated workflow

### Known Issues

- release tagging still needs to be completed on GitHub
- packaging metadata is minimal and intended for course-deliverable readiness rather than PyPI publication
- some presentation artifacts and large generated datasets should remain outside the Git repository
