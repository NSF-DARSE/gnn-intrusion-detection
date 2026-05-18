# Project Onboarding

This note is for the next student or teammate who needs to understand the
repository quickly.

## First Read

If you are new to this repo, read in this order:

1. `README.md`
2. `docs/module_a_documentation.md`
3. `docs/module_b_documentation.md`
4. `docs/module_c_documentation.md`

Then open the main code files:

1. `gcn_ids/data_graph.py`
2. `gcn_ids/learning.py`
3. `gcn_ids/graph_viz.py`
4. `main.py`

## The Main Pipeline

The project is divided into three modules:

- **Module A**
  - cleans raw flow data
  - groups flows into fixed windows
  - builds graph artifacts

- **Module B**
  - loads graph artifacts
  - trains and evaluates the GCN
  - writes metrics and plots

- **Module C**
  - reads Module B predictions
  - renders test-set graph visualizations

## The Most Important Folders

- `gcn_ids/`
  - core code for the three modules
- `docs/`
  - human-readable documentation
- `scripts/`
  - helper scripts
- `tests/`
  - current automated tests

## Which Dataset To Use

For the main workflow, use:

- `UNSW-NB15`

The latest graph build used in this repo is:

- `data/graph_unsw_full_10min_stratified_clean/`

The IoT-style dataset is still present in the repo history, but it is secondary.

## Expected Workflow For A New Contributor

If you are continuing development:

1. set up the environment
2. run the Module A tests
3. inspect the existing UNSW graph build
4. run Module B and Module C on that build
5. only then start changing code

## Current Testing Situation

The best automated test coverage is for **Module A**:

- `tests/test_data_graph.py`

Module B and Module C are currently validated more through:

- saved outputs
- plots
- integration runs

So if you extend Module B or Module C, adding dedicated tests there would be a
good next step.

## Important Practical Note

This repository was cleaned to be easier to understand, but the project itself
went through multiple phases. That means:

- some older local artifacts may still exist outside the tracked repo structure
- the presentation used by the team may not perfectly match the latest repo
  organization

If you are ever unsure, trust the current code and the current README over old
presentation language.
