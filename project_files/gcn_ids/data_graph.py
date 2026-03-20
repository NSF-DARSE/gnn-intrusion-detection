#!/usr/bin/env python3
"""Module A pipeline: clean IDS flows, window them, and export graph artifacts.

This module builds per-window static directed graphs for node classification:
- Nodes: IP addresses
- Edges: flows within a fixed time window
- Node labels: flow labels are evidence for both source and destination nodes
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("data_graph")


CANONICAL_COLUMNS = {
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "binary_label",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Average Packet Size",
}

ALIASES = {
    "src ip": "Source IP",
    "source ip": "Source IP",
    "id orig h": "Source IP",
    "id.orig_h": "Source IP",
    "dst ip": "Destination IP",
    "destination ip": "Destination IP",
    "id resp h": "Destination IP",
    "id.resp_h": "Destination IP",
    "timestamp": "Timestamp",
    "ts": "Timestamp",
    "label": "Label",
    "binary label": "binary_label",
    "binary_label": "binary_label",
    "protocol": "Protocol",
    "proto": "Protocol",
    "flow duration": "Flow Duration",
    "duration": "Flow Duration",
    "total fwd packets": "Total Fwd Packets",
    "tot fwd pkts": "Total Fwd Packets",
    "orig pkts": "Total Fwd Packets",
    "orig_pkts": "Total Fwd Packets",
    "total backward packets": "Total Backward Packets",
    "tot bwd pkts": "Total Backward Packets",
    "resp pkts": "Total Backward Packets",
    "resp_pkts": "Total Backward Packets",
    "total length of fwd packets": "Total Length of Fwd Packets",
    "totlen fwd pkts": "Total Length of Fwd Packets",
    "orig ip bytes": "Total Length of Fwd Packets",
    "orig_ip_bytes": "Total Length of Fwd Packets",
    "orig bytes": "Total Length of Fwd Packets",
    "orig_bytes": "Total Length of Fwd Packets",
    "total length of bwd packets": "Total Length of Bwd Packets",
    "totlen bwd pkts": "Total Length of Bwd Packets",
    "resp ip bytes": "Total Length of Bwd Packets",
    "resp_ip_bytes": "Total Length of Bwd Packets",
    "resp bytes": "Total Length of Bwd Packets",
    "resp_bytes": "Total Length of Bwd Packets",
    "flow bytes/s": "Flow Bytes/s",
    "flow byts/s": "Flow Bytes/s",
    "flow packets/s": "Flow Packets/s",
    "flow pkts/s": "Flow Packets/s",
    "average packet size": "Average Packet Size",
    "pkt size avg": "Average Packet Size",
}

PROTOCOL_MAP = {
    "tcp": 6,
    "udp": 17,
    "icmp": 1,
}

NODE_FEATURES = [
    "flow_count",
    "malicious_flow_ratio",
    "inbound_flow_count",
    "outbound_flow_count",
    "unique_peer_count",
    "protocol_tcp_ratio",
    "protocol_udp_ratio",
    "protocol_other_ratio",
    "total_packets_sum",
    "total_bytes_sum",
    "avg_flow_duration",
    "avg_flow_bytes_per_s",
    "avg_flow_packets_per_s",
    "avg_packet_size",
]

EDGE_FEATURES = [
    "flow_count",
    "malicious_ratio",
    "total_packets_sum",
    "total_bytes_sum",
    "avg_flow_duration",
    "avg_flow_bytes_per_s",
    "avg_flow_packets_per_s",
    "protocol_tcp_ratio",
    "protocol_udp_ratio",
    "protocol_other_ratio",
]


@dataclass
class NodeAgg:
    flow_count: int = 0
    malicious_count: int = 0
    benign_count: int = 0
    inbound_flow_count: int = 0
    outbound_flow_count: int = 0
    protocol_tcp_count: int = 0
    protocol_udp_count: int = 0
    protocol_other_count: int = 0
    total_packets_sum: float = 0.0
    total_bytes_sum: float = 0.0
    flow_duration_sum: float = 0.0
    flow_bytes_per_s_sum: float = 0.0
    flow_packets_per_s_sum: float = 0.0
    avg_packet_size_sum: float = 0.0


@dataclass
class EdgeAgg:
    flow_count: int = 0
    malicious_count: int = 0
    total_packets_sum: float = 0.0
    total_bytes_sum: float = 0.0
    flow_duration_sum: float = 0.0
    flow_bytes_per_s_sum: float = 0.0
    flow_packets_per_s_sum: float = 0.0
    protocol_tcp_count: int = 0
    protocol_udp_count: int = 0
    protocol_other_count: int = 0


def _normalize_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    return key


def canonicalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for c in columns:
        key = _normalize_key(c)
        canonical = ALIASES.get(key, c)
        mapping[c] = canonical
    return mapping


def safe_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values.fillna(0.0).astype(float)


def parse_protocol(series: pd.Series) -> pd.Series:
    as_num = pd.to_numeric(series, errors="coerce")
    mask_num = as_num.notna()
    output = pd.Series(np.zeros(len(series), dtype=np.int64), index=series.index)
    output.loc[mask_num] = as_num.loc[mask_num].astype(np.int64)

    as_text = series.astype(str).str.lower().str.strip()
    for token, proto_id in PROTOCOL_MAP.items():
        output.loc[(~mask_num) & (as_text == token)] = proto_id
    return output


def parse_binary_label(series: pd.Series) -> pd.Series:
    tokens = series.astype(str).str.strip().str.lower()
    benign_tokens = {"0", "benign", "normal", "background"}
    malicious_tokens = {"1", "malicious", "attack", "anomaly", "botnet", "ddos", "dos"}
    output = pd.Series(np.ones(len(series), dtype=np.int32), index=series.index)
    output.loc[tokens.isin(benign_tokens)] = 0
    output.loc[tokens.isin(malicious_tokens)] = 1
    return output


def parse_timestamps(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    sec_mask = numeric.notna() & (numeric < 1e12)
    if sec_mask.any():
        parsed.loc[sec_mask] = pd.to_datetime(numeric.loc[sec_mask], unit="s", errors="coerce", utc=True).dt.tz_localize(None)

    ms_mask = numeric.notna() & (numeric >= 1e12)
    if ms_mask.any():
        parsed.loc[ms_mask] = pd.to_datetime(numeric.loc[ms_mask], unit="ms", errors="coerce", utc=True).dt.tz_localize(None)

    text_mask = numeric.isna()
    if text_mask.any():
        parsed.loc[text_mask] = pd.to_datetime(series.loc[text_mask], errors="coerce", utc=False)
    return parsed


def parse_window_to_minutes(window_size: int, window_unit: str, window: Optional[str]) -> int:
    if window is not None:
        m = re.fullmatch(r"\s*(\d+)\s*(m|min|mins|minute|minutes|h|hr|hour|hours)\s*", window.lower())
        if not m:
            raise ValueError(f"Invalid --window value: {window}")
        size = int(m.group(1))
        unit = m.group(2)
    else:
        size = window_size
        unit = window_unit.lower()

    if size <= 0:
        raise ValueError("Window size must be > 0.")

    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return size
    if unit in {"h", "hr", "hour", "hours"}:
        return size * 60
    raise ValueError(f"Unsupported window unit: {window_unit}")


def parse_split_ratio(split_ratio: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in split_ratio.split(",")]
    if len(parts) != 3:
        raise ValueError("--split-ratio must contain exactly 3 comma-separated values.")
    train_r, val_r, test_r = parts
    if min(parts) <= 0:
        raise ValueError("All split ratio values must be positive.")
    total = train_r + val_r + test_r
    train_r /= total
    val_r /= total
    test_r /= total
    return train_r, val_r, test_r


def assign_temporal_splits(windows: Sequence[pd.Timestamp], split_ratio: Tuple[float, float, float]) -> Dict[pd.Timestamp, str]:
    train_r, val_r, _test_r = split_ratio
    n = len(windows)
    train_end = int(n * train_r)
    val_end = int(n * (train_r + val_r))
    split_map: Dict[pd.Timestamp, str] = {}
    for idx, window_start in enumerate(windows):
        if idx < train_end:
            split_map[window_start] = "train"
        elif idx < val_end:
            split_map[window_start] = "val"
        else:
            split_map[window_start] = "test"
    return split_map


def load_and_clean_csvs(
    input_files: Sequence[Path],
    max_rows_per_file: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    cleaned_frames: List[pd.DataFrame] = []
    used_files: List[Dict[str, object]] = []
    skipped_files: List[Dict[str, object]] = []

    for file_path in input_files:
        if not file_path.exists():
            skipped_files.append({"file": str(file_path), "reason": "not_found"})
            continue

        df = pd.read_csv(file_path, low_memory=False)
        original_rows = len(df)
        rename_map = canonicalize_columns(df.columns)
        df = df.rename(columns=rename_map)

        required = ["Source IP", "Destination IP", "Timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            skipped_files.append({"file": str(file_path), "reason": "missing_required_columns", "missing": missing})
            continue

        if max_rows_per_file is not None and len(df) > max_rows_per_file:
            keep_idx = rng.choice(df.index.values, size=max_rows_per_file, replace=False)
            df = df.loc[np.sort(keep_idx)].copy()

        if "Label" not in df.columns:
            if "binary_label" in df.columns:
                df["Label"] = df["binary_label"].astype(str)
            else:
                df["Label"] = "Unknown"

        if "binary_label" in df.columns:
            bin_series = parse_binary_label(df["binary_label"])
        else:
            bin_series = parse_binary_label(df["Label"])
        df["binary_label"] = bin_series.astype(int)

        numeric_defaults = [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Average Packet Size",
        ]
        for col in numeric_defaults:
            if col not in df.columns:
                df[col] = 0.0

        df["Protocol"] = parse_protocol(df["Protocol"] if "Protocol" in df.columns else pd.Series(["0"] * len(df)))
        for col in numeric_defaults:
            df[col] = safe_numeric(df[col])

        total_packets = df["Total Fwd Packets"] + df["Total Backward Packets"]
        total_bytes = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
        duration = df["Flow Duration"].replace(0.0, np.nan)

        flow_bps_missing = df["Flow Bytes/s"] == 0.0
        df.loc[flow_bps_missing, "Flow Bytes/s"] = (total_bytes / duration).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        flow_pps_missing = df["Flow Packets/s"] == 0.0
        df.loc[flow_pps_missing, "Flow Packets/s"] = (total_packets / duration).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        avg_pkt_missing = df["Average Packet Size"] == 0.0
        df.loc[avg_pkt_missing, "Average Packet Size"] = (total_bytes / total_packets.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df["Timestamp"] = parse_timestamps(df["Timestamp"])
        before_drop = len(df)
        df = df.dropna(subset=["Timestamp"]).copy()
        dropped = before_drop - len(df)
        if len(df) == 0:
            skipped_files.append({"file": str(file_path), "reason": "no_valid_timestamps"})
            continue

        df["Source IP"] = df["Source IP"].astype(str).str.strip()
        df["Destination IP"] = df["Destination IP"].astype(str).str.strip()
        df = df[(df["Source IP"] != "") & (df["Destination IP"] != "")].copy()

        df["source_file"] = file_path.name

        keep_cols = [
            "source_file",
            "Source IP",
            "Destination IP",
            "Timestamp",
            "Label",
            "binary_label",
            "Protocol",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Average Packet Size",
        ]
        df = df[keep_cols].copy()
        cleaned_frames.append(df)

        used_files.append(
            {
                "file": str(file_path),
                "original_rows": original_rows,
                "rows_after_sampling": int(len(df) + dropped),
                "rows_after_cleaning": int(len(df)),
                "dropped_invalid_timestamps": int(dropped),
                "benign_rows": int((df["binary_label"] == 0).sum()),
                "malicious_rows": int((df["binary_label"] == 1).sum()),
            }
        )

    if not cleaned_frames:
        raise ValueError("No valid rows loaded from input files.")

    merged = pd.concat(cleaned_frames, ignore_index=True)
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    return merged, used_files, skipped_files


def _update_node_agg(agg: NodeAgg, row: pd.Series, is_source: bool) -> None:
    proto = int(row["Protocol"])
    is_mal = int(row["binary_label"]) == 1
    total_fwd = float(row["Total Fwd Packets"])
    total_bwd = float(row["Total Backward Packets"])
    total_fwd_bytes = float(row["Total Length of Fwd Packets"])
    total_bwd_bytes = float(row["Total Length of Bwd Packets"])

    agg.flow_count += 1
    agg.malicious_count += int(is_mal)
    agg.benign_count += int(not is_mal)
    agg.outbound_flow_count += int(is_source)
    agg.inbound_flow_count += int(not is_source)

    if proto == 6:
        agg.protocol_tcp_count += 1
    elif proto == 17:
        agg.protocol_udp_count += 1
    else:
        agg.protocol_other_count += 1

    agg.total_packets_sum += total_fwd + total_bwd
    agg.total_bytes_sum += total_fwd_bytes + total_bwd_bytes
    agg.flow_duration_sum += float(row["Flow Duration"])
    agg.flow_bytes_per_s_sum += float(row["Flow Bytes/s"])
    agg.flow_packets_per_s_sum += float(row["Flow Packets/s"])
    agg.avg_packet_size_sum += float(row["Average Packet Size"])


def _update_edge_agg(agg: EdgeAgg, row: pd.Series) -> None:
    proto = int(row["Protocol"])
    is_mal = int(row["binary_label"]) == 1
    total_fwd = float(row["Total Fwd Packets"])
    total_bwd = float(row["Total Backward Packets"])
    total_fwd_bytes = float(row["Total Length of Fwd Packets"])
    total_bwd_bytes = float(row["Total Length of Bwd Packets"])

    agg.flow_count += 1
    agg.malicious_count += int(is_mal)
    agg.total_packets_sum += total_fwd + total_bwd
    agg.total_bytes_sum += total_fwd_bytes + total_bwd_bytes
    agg.flow_duration_sum += float(row["Flow Duration"])
    agg.flow_bytes_per_s_sum += float(row["Flow Bytes/s"])
    agg.flow_packets_per_s_sum += float(row["Flow Packets/s"])

    if proto == 6:
        agg.protocol_tcp_count += 1
    elif proto == 17:
        agg.protocol_udp_count += 1
    else:
        agg.protocol_other_count += 1


def build_window_graph_arrays(
    window_df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    label_rule: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = len(node_to_idx)
    node_aggs: Dict[int, NodeAgg] = defaultdict(NodeAgg)
    node_peers: Dict[int, set] = defaultdict(set)
    edge_aggs: Dict[Tuple[int, int], EdgeAgg] = defaultdict(EdgeAgg)

    for row in window_df.itertuples(index=False):
        src_ip = getattr(row, "Source_IP")
        dst_ip = getattr(row, "Destination_IP")
        src_idx = node_to_idx[src_ip]
        dst_idx = node_to_idx[dst_ip]

        row_series = pd.Series(
            {
                "Protocol": getattr(row, "Protocol"),
                "binary_label": getattr(row, "binary_label"),
                "Total Fwd Packets": getattr(row, "Total_Fwd_Packets"),
                "Total Backward Packets": getattr(row, "Total_Backward_Packets"),
                "Total Length of Fwd Packets": getattr(row, "Total_Length_of_Fwd_Packets"),
                "Total Length of Bwd Packets": getattr(row, "Total_Length_of_Bwd_Packets"),
                "Flow Duration": getattr(row, "Flow_Duration"),
                "Flow Bytes/s": getattr(row, "Flow_Bytes_s"),
                "Flow Packets/s": getattr(row, "Flow_Packets_s"),
                "Average Packet Size": getattr(row, "Average_Packet_Size"),
            }
        )

        _update_node_agg(node_aggs[src_idx], row_series, is_source=True)
        _update_node_agg(node_aggs[dst_idx], row_series, is_source=False)
        _update_edge_agg(edge_aggs[(src_idx, dst_idx)], row_series)
        node_peers[src_idx].add(dst_idx)
        node_peers[dst_idx].add(src_idx)

    node_features = np.zeros((num_nodes, len(NODE_FEATURES)), dtype=np.float32)
    node_labels = np.zeros((num_nodes,), dtype=np.int8)

    for idx, agg in node_aggs.items():
        flow_count = max(agg.flow_count, 1)
        if label_rule == "any_malicious_wins":
            node_labels[idx] = 1 if agg.malicious_count > 0 else 0
        elif label_rule == "majority":
            node_labels[idx] = 1 if agg.malicious_count > agg.benign_count else 0
        else:
            raise ValueError(f"Unsupported label rule: {label_rule}")

        tcp_ratio = agg.protocol_tcp_count / flow_count
        udp_ratio = agg.protocol_udp_count / flow_count
        other_ratio = agg.protocol_other_count / flow_count

        node_features[idx] = np.array(
            [
                float(agg.flow_count),
                float(agg.malicious_count / flow_count),
                float(agg.inbound_flow_count),
                float(agg.outbound_flow_count),
                float(len(node_peers[idx])),
                float(tcp_ratio),
                float(udp_ratio),
                float(other_ratio),
                float(agg.total_packets_sum),
                float(agg.total_bytes_sum),
                float(agg.flow_duration_sum / flow_count),
                float(agg.flow_bytes_per_s_sum / flow_count),
                float(agg.flow_packets_per_s_sum / flow_count),
                float(agg.avg_packet_size_sum / flow_count),
            ],
            dtype=np.float32,
        )

    if edge_aggs:
        edge_index = np.zeros((2, len(edge_aggs)), dtype=np.int32)
        edge_features = np.zeros((len(edge_aggs), len(EDGE_FEATURES)), dtype=np.float32)
        for e_idx, ((src_idx, dst_idx), agg) in enumerate(sorted(edge_aggs.items(), key=lambda x: (x[0][0], x[0][1]))):
            flow_count = max(agg.flow_count, 1)
            edge_index[:, e_idx] = np.array([src_idx, dst_idx], dtype=np.int32)
            edge_features[e_idx] = np.array(
                [
                    float(agg.flow_count),
                    float(agg.malicious_count / flow_count),
                    float(agg.total_packets_sum),
                    float(agg.total_bytes_sum),
                    float(agg.flow_duration_sum / flow_count),
                    float(agg.flow_bytes_per_s_sum / flow_count),
                    float(agg.flow_packets_per_s_sum / flow_count),
                    float(agg.protocol_tcp_count / flow_count),
                    float(agg.protocol_udp_count / flow_count),
                    float(agg.protocol_other_count / flow_count),
                ],
                dtype=np.float32,
            )
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)
        edge_features = np.zeros((0, len(EDGE_FEATURES)), dtype=np.float32)

    return node_features, node_labels, edge_index, edge_features


def fit_scalers(train_graphs: List[Dict[str, np.ndarray]]) -> Tuple[StandardScaler, Optional[StandardScaler]]:
    node_batches = []
    edge_batches = []

    for graph in train_graphs:
        node_x = graph["node_features"]
        active_mask = node_x[:, 0] > 0.0
        if active_mask.any():
            node_batches.append(node_x[active_mask])
        edge_x = graph["edge_features"]
        if len(edge_x) > 0:
            edge_batches.append(edge_x)

    node_scaler = StandardScaler()
    node_fit = np.vstack(node_batches) if node_batches else np.zeros((1, len(NODE_FEATURES)), dtype=np.float32)
    node_scaler.fit(node_fit)

    edge_scaler: Optional[StandardScaler]
    if edge_batches:
        edge_scaler = StandardScaler()
        edge_scaler.fit(np.vstack(edge_batches))
    else:
        edge_scaler = None

    return node_scaler, edge_scaler


def apply_scalers(
    graph: Dict[str, np.ndarray],
    node_scaler: StandardScaler,
    edge_scaler: Optional[StandardScaler],
) -> Dict[str, np.ndarray]:
    scaled = dict(graph)
    scaled["node_features"] = node_scaler.transform(graph["node_features"]).astype(np.float32)
    if edge_scaler is not None and len(graph["edge_features"]) > 0:
        scaled["edge_features"] = edge_scaler.transform(graph["edge_features"]).astype(np.float32)
    else:
        scaled["edge_features"] = graph["edge_features"].astype(np.float32)
    return scaled


def save_graph_npz(path: Path, graph: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        node_features=graph["node_features"],
        node_labels=graph["node_labels"],
        edge_index=graph["edge_index"],
        edge_features=graph["edge_features"],
    )


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    input_files: List[Path] = []
    for item in args.input_files:
        p = Path(item)
        if p.is_dir():
            input_files.extend(sorted(p.glob(args.input_glob)))
        else:
            input_files.append(p)
    input_files = sorted(set(input_files))
    if not input_files:
        raise ValueError("No input files found.")

    window_minutes = parse_window_to_minutes(args.window_size, args.window_unit, args.window)
    split_ratio = parse_split_ratio(args.split_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flows, used_files, skipped_files = load_and_clean_csvs(
        input_files=input_files,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed,
    )

    flows["window_start"] = flows["Timestamp"].dt.floor(f"{window_minutes}min")
    windows = list(flows["window_start"].sort_values().drop_duplicates())
    split_map = assign_temporal_splits(windows, split_ratio)

    all_ips = sorted(set(flows["Source IP"]).union(set(flows["Destination IP"])))
    node_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}

    # Replace spaces and slashes for itertuples named attributes.
    tuple_ready = flows.rename(
        columns={
            "Source IP": "Source_IP",
            "Destination IP": "Destination_IP",
            "Total Fwd Packets": "Total_Fwd_Packets",
            "Total Backward Packets": "Total_Backward_Packets",
            "Total Length of Fwd Packets": "Total_Length_of_Fwd_Packets",
            "Total Length of Bwd Packets": "Total_Length_of_Bwd_Packets",
            "Flow Duration": "Flow_Duration",
            "Flow Bytes/s": "Flow_Bytes_s",
            "Flow Packets/s": "Flow_Packets_s",
            "Average Packet Size": "Average_Packet_Size",
        }
    )

    window_graphs: List[Dict[str, object]] = []
    for w in windows:
        w_df = tuple_ready[tuple_ready["window_start"] == w]
        node_x, node_y, edge_idx, edge_x = build_window_graph_arrays(
            w_df,
            node_to_idx=node_to_idx,
            label_rule=args.node_label_rule,
        )
        split_name = split_map[w]
        window_graphs.append(
            {
                "window_start": w,
                "split": split_name,
                "node_features": node_x,
                "node_labels": node_y,
                "edge_index": edge_idx,
                "edge_features": edge_x,
            }
        )

    train_graphs = [g for g in window_graphs if g["split"] == "train"]
    if not train_graphs:
        raise ValueError("Train split is empty. Adjust split ratio or input size.")

    node_scaler, edge_scaler = fit_scalers(train_graphs)

    split_counts = {"train": 0, "val": 0, "test": 0}
    graph_meta = []
    for g_idx, graph in enumerate(window_graphs):
        scaled_graph = apply_scalers(graph, node_scaler=node_scaler, edge_scaler=edge_scaler)
        split_name = scaled_graph["split"]
        split_counts[split_name] += 1
        file_name = f"window_{g_idx:05d}.npz"
        out_path = output_dir / "graphs" / split_name / file_name
        save_graph_npz(out_path, scaled_graph)

        graph_meta.append(
            {
                "file": str(out_path),
                "split": split_name,
                "window_start": str(pd.Timestamp(scaled_graph["window_start"])),
                "num_nodes": int(scaled_graph["node_features"].shape[0]),
                "num_edges": int(scaled_graph["edge_features"].shape[0]),
                "node_feature_dim": int(scaled_graph["node_features"].shape[1]),
                "edge_feature_dim": int(scaled_graph["edge_features"].shape[1]),
            }
        )

    if args.save_cleaned_flows:
        cleaned_csv = output_dir / "merged_cleaned_flows.csv"
        flows.to_csv(cleaned_csv, index=False)
    else:
        cleaned_csv = None

    joblib.dump(node_scaler, output_dir / "scaler_node.pkl")
    if edge_scaler is not None:
        joblib.dump(edge_scaler, output_dir / "scaler_edge.pkl")

    schema = {
        "node_features": NODE_FEATURES,
        "edge_features": EDGE_FEATURES,
        "label_mapping": {"benign": 0, "malicious": 1},
        "node_id_mapping_file": "node_mapping.json",
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output_dir / "node_mapping.json").write_text(json.dumps(node_to_idx, indent=2), encoding="utf-8")

    manifest = {
        "pipeline": "gcn_ids.data_graph",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "window_minutes": window_minutes,
            "window_size_arg": args.window_size,
            "window_unit_arg": args.window_unit,
            "window_arg": args.window,
            "node_label_rule": args.node_label_rule,
            "split_ratio": {
                "train": split_ratio[0],
                "val": split_ratio[1],
                "test": split_ratio[2],
            },
            "seed": args.seed,
            "input_glob": args.input_glob,
            "max_rows_per_file": args.max_rows_per_file,
        },
        "inputs": {
            "requested_files": [str(x) for x in input_files],
            "used_files": used_files,
            "skipped_files": skipped_files,
        },
        "summary": {
            "rows_total_cleaned": int(len(flows)),
            "binary_label_distribution": {
                "benign_0": int((flows["binary_label"] == 0).sum()),
                "malicious_1": int((flows["binary_label"] == 1).sum()),
            },
            "num_nodes_global": int(len(node_to_idx)),
            "num_windows_total": int(len(windows)),
            "graphs_per_split": split_counts,
            "feature_dims": {
                "node": len(NODE_FEATURES),
                "edge": len(EDGE_FEATURES),
            },
        },
        "artifacts": {
            "graphs_dir": str(output_dir / "graphs"),
            "cleaned_flows_csv": str(cleaned_csv) if cleaned_csv else None,
            "node_scaler": str(output_dir / "scaler_node.pkl"),
            "edge_scaler": str(output_dir / "scaler_edge.pkl") if edge_scaler is not None else None,
            "schema": str(output_dir / "schema.json"),
            "node_mapping": str(output_dir / "node_mapping.json"),
        },
        "scaling": {
            "fit_on": "train_only",
            "note": "Node scaler fit on active train nodes; edge scaler fit on train edges only.",
        },
        "graphs": graph_meta,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build windowed graph dataset artifacts for IDS node classification.")
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="One or more CSV files and/or directories containing CSV files.",
    )
    parser.add_argument("--input-glob", default="*.csv", help="Glob pattern when an input path is a directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for graph artifacts.")
    parser.add_argument("--window-size", type=int, default=10, help="Window size value.")
    parser.add_argument(
        "--window-unit",
        default="minutes",
        choices=["m", "min", "mins", "minute", "minutes", "h", "hr", "hour", "hours"],
        help="Window unit if --window is not provided.",
    )
    parser.add_argument("--window", default=None, help='Compact window format, e.g. "10min" or "1h".')
    parser.add_argument(
        "--node-label-rule",
        choices=["any_malicious_wins", "majority"],
        default="any_malicious_wins",
        help="Node label rule from flow evidence in each window.",
    )
    parser.add_argument(
        "--split-ratio",
        default="0.6,0.2,0.2",
        help="Temporal split ratio for train,val,test.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional random cap per input file for faster experiments.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--save-cleaned-flows",
        action="store_true",
        help="If set, save merged cleaned flow CSV to output folder.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    manifest = run_pipeline(args)
    LOGGER.info(
        "Completed graph build: windows=%s train=%s val=%s test=%s",
        manifest["summary"]["num_windows_total"],
        manifest["summary"]["graphs_per_split"]["train"],
        manifest["summary"]["graphs_per_split"]["val"],
        manifest["summary"]["graphs_per_split"]["test"],
    )


if __name__ == "__main__":
    main()
