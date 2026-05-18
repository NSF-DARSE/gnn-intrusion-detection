# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-05-05

### Added

- Module A graph-building pipeline in `gcn_ids/data_graph.py`
- Module B training and evaluation pipeline in `gcn_ids/learning.py`
- Module C test-set graph visualization pipeline in `gcn_ids/graph_viz.py`
- integrated workflow entry point in `main.py`
- Module A documentation and handoff notes in `docs/`
- Module A tests in `tests/test_data_graph.py`
- stratified window splitting by attack presence for imbalanced data handling
- UNSW raw headerless shard support
- confusion matrix and training-curve generation for Module B
- release notes and packaging metadata files

### Changed

- updated the project from earlier IoT-only experimentation to support the mentor-selected UNSW-NB15 dataset
- improved dataset splitting from chronological-only support to optional stratified splitting
- updated the top-level README to describe the full three-module workflow

### Fixed

- removed reliance on hard-coded Mac-specific dataset paths
- normalized raw UNSW field aliases into the project schema
- made Module A handoff easier for Windows users with a PowerShell runner

### Known Limitations

- the repo does not yet include a published GitHub release tag
- current test coverage is strongest for Module A and lighter for Modules B and C
- generated datasets and model outputs are large and are not intended to be committed in full to GitHub
