# Case Study: Network Intrusion Detection and Visualization Pipeline

## Overview
**Problem:** Identifying malicious network traffic in real-time is highly complex due to the massive, interconnected nature of network data. Traditional security visualization tools often create unreadable "hairball" graphs, providing little actionable intelligence to security teams.
**Stakeholders:** Security Analysts, Network Administrators, and Data Science Researchers.
**Solution:** An end-to-end Graph Neural Network (GNN) pipeline. It combines a machine learning backend to detect anomalies with a high-speed, multi-core visualization frontend. The visualization engine uses a mathematically stable "Multi-Shell" layout to generate clear, publication-ready network graphs that automatically map model performance (True Positives, False Positives, etc.) to distinct shapes and colors.

## Repository Structure
- `src/` – source code
  - `gnn-intrusion-detection/project_files/gcn_ids/baselines_viz.py` – The Multi-Shell visualization and rendering engine.
  - `gnn-intrusion-detection/project_files/gcn_ids/learning.py` – The GNN model training and prediction inference engine.
- `docs/` – optional documentation (Sphinx scaffold) and generated model metrics text files.
- `data/` – input/output data (if applicable)
  - `/graphs/test/` – Zipped adjacency matrices (`.npz`) representing time-windowed network connections.
  - Raw NumPy arrays containing model predictions and labels (`.npy`).

## Getting Started
1. **Clone the repository:** Download the project files to your local or shared workspace.
2. **Create a feature branch:** Always branch off main for your individual work (e.g., `git checkout -b feature/update-viz`).
3. **Run the Pipeline (Locally/Workspace):**
   - *Backend:* `python gnn-intrusion-detection/project_files/gcn_ids/learning.py --dataset ids-custom --dataset_dir data/`
   - *Frontend:* `python gnn-intrusion-detection/project_files/gcn_ids/baselines_viz.py --dataset ids-custom --dataset_dir data/`
4. **Open a pull request early:** Submit your PR as soon as you have a working draft to allow for team review and collaboration.

## Documentation
This repository includes an optional Sphinx documentation scaffold. 
Additionally, the visual outputs from `baselines_viz.py` are self-documenting. Every generated image includes dynamic live counts (total nodes, active nodes, active connections) and a performance legend (🔴 TP, 🟦 TN, ⚠️ FP, 🟢 FN) to clearly explain the network state at any given fraction of a second.

## Contributing
All changes must go through pull requests. Please ensure any updates to the visualization or ML scripts are tested against the `.npz` network graph format before requesting a review.
