# GCN-IDS: Graph Convolutional Networks for Network Intrusion Detection

A graph-based machine learning system for detecting malicious network traffic using Graph Neural Networks (GNNs). The system models network flows as temporal graphs and uses GCN/GAT/GraphSAGE architectures for node classification.

## Overview

**Problem:** Traditional network intrusion detection systems (IDS) rely on signature-based detection that can be easily evaded by novel attacks.

**Solution:** GCN-IDS represents network communication as graphs where:
- **Nodes:** IP addresses (hosts)
- **Edges:** Network flows between IPs
- **Labels:** Benign (0) vs Malicious (1)

This graph representation captures communication patterns that are harder to forge than individual flow signatures.

## Project Structure

```
.
├── project_files/
│   └── gcn_ids/
│       ├── data_graph.py           # Data pipeline: CSV → Windowed Graphs
│       ├── learning.py             # Training: GCN/GAT/GraphSAGE training
│       ├── baselines_viz.py        # Visualization: Network graphs with predictions
│       └── graph_10min_moduleA/    # Pre-generated graph artifacts
│           ├── graphs/
│           │   ├── train/          # 150 windows
│           │   ├── val/            # 50 windows
│           │   └── test/           # 50 windows
│           ├── manifest.json       # Pipeline metadata
│           └── schema.json         # Feature definitions
├── test_data_graph.py              # Unit tests
└── requirements.txt                # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- joblib>=1.3.0
- pytest>=7.0.0
- torch>=2.0.0
- torch-geometric

## Quick Start

### 1. Data Pipeline

Convert raw CSV network flows to windowed graph artifacts:

```bash
python -m project_files.gcn_ids.data_graph \
  --input-files /path/to/flows.csv \
  --output-dir ./graph_output \
  --window 10min \
  --split-ratio 0.6,0.2,0.2 \
  --node-label-rule any_malicious_wins
```

**Input CSV Requirements:**
- Source IP, Destination IP, Timestamp (required)
- Protocol, Flow Duration, Packet counts (optional - imputed if missing)
- Label or binary_label column

**Output:**
- `graphs/{train,val,test}/window_*.npz` - Graph artifacts
- `manifest.json` - Full pipeline provenance
- `scaler_*.pkl` - Fitted StandardScalers

### 2. Training

Train a GNN model on the graph dataset:

```bash
python project_files/gcn_ids/learning.py \
  --dataset ids-custom \
  --dataset_dir ./graph_output \
  --model GCN \
  --epochs 50 \
  --hidden_channels 256 \
  --lr 0.003
```

**Model Options:** `GCN`, `GAT`, `SAGE`

**Output:**
- `confusion_matrix_{dataset}_{epochs}epochs.png`
- `metrics_{dataset}_{epochs}epochs.txt`
- `predictions_{dataset}/window_*_preds.npy`

### 3. Visualization

Generate network visualizations with classification results:

```bash
python project_files/gcn_ids/baselines_viz.py \
  --dataset ids-custom \
  --dataset_dir ./graph_output
```

## Features

### Node Features (14 dimensions)
1. `flow_count` - Total flows involving this IP
2. `malicious_flow_ratio` - Proportion of malicious flows
3. `inbound_flow_count` - Flows received
4. `outbound_flow_count` - Flows sent
5. `unique_peer_count` - Unique communicating IPs
6. `protocol_tcp_ratio` - TCP flow proportion
7. `protocol_udp_ratio` - UDP flow proportion
8. `protocol_other_ratio` - Other protocol proportion
9. `total_packets_sum` - Total packet count
10. `total_bytes_sum` - Total byte count
11. `avg_flow_duration` - Mean flow duration
12. `avg_flow_bytes_per_s` - Mean bytes/second
13. `avg_flow_packets_per_s` - Mean packets/second
14. `avg_packet_size` - Mean packet size

### Edge Features (10 dimensions)
1. `flow_count` - Flows between this pair
2. `malicious_ratio` - Proportion of malicious flows
3. `total_packets_sum` - Total packets
4. `total_bytes_sum` - Total bytes
5. `avg_flow_duration` - Mean duration
6. `avg_flow_bytes_per_s` - Mean bytes/second
7. `avg_flow_packets_per_s` - Mean packets/second
8. `protocol_tcp_ratio` - TCP proportion
9. `protocol_udp_ratio` - UDP proportion
10. `protocol_other_ratio` - Other protocol proportion

## Testing

Run the test suite:

```bash
pytest project_files/test_data_graph.py -v
```

**Test Coverage:**
- Window duration parsing
- Split ratio normalization
- Temporal split ordering
- Binary label mapping
- Pipeline artifact generation

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Data Pipeline                      │
├─────────────────────────────────────────────────────┤
│  Raw CSV → Clean → Window → Features → NPZ         │
│                                                     │
│  Temporal 10-min windows                            │
│  60/20/20 train/val/test split                     │
│  StandardScaler fit on training only               │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                Training Pipeline                    │
├─────────────────────────────────────────────────────┤
│  NeighborLoader → GCN/GAT/SAGE → Classification    │
│                                                     │
│  Binary cross-entropy loss                         │
│  Adam optimizer with weight decay                  │
│  GPU acceleration with RMM memory management       │
└─────────────────────────────────────────────────────┘
```

## GPU Support

The system supports CUDA GPUs with RMM memory pooling:

```python
# Automatic if CUDA available
rmm.reinitialize(devices=[0], pool_allocator=True, managed_memory=True)
cupy.cuda.set_allocator(rmm_cupy_allocator)
```

For distributed training:
```bash
export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

## Dataset Statistics

Pre-generated dataset (`graph_10min_moduleA`):
- **Total flows:** 120,000 cleaned records
- **Label distribution:** 96,938 benign (80.8%), 23,062 malicious (19.2%)
- **Unique IPs:** 9,444 per window
- **Windows:** 250 total (10-minute intervals)
- **Temporal range:** March-May 2017

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Release history
- [LICENSE](LICENSE) - MIT License

## Contributing

All changes must go through pull requests. See project guidelines for details.

## License

MIT License - Copyright (c) 2026 NSF-DARSE

## Acknowledgments

Built with PyTorch Geometric for graph neural network operations.