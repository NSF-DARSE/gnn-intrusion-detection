from pathlib import Path

import pandas as pd

from gcn_ids.data_graph import (
    assign_temporal_splits,
    parse_binary_label,
    parse_split_ratio,
    parse_window_to_minutes,
    run_pipeline,
)


def test_parse_window_to_minutes():
    assert parse_window_to_minutes(5, "minutes", None) == 5
    assert parse_window_to_minutes(1, "hours", None) == 60
    assert parse_window_to_minutes(10, "minutes", "10min") == 10
    assert parse_window_to_minutes(10, "minutes", "2h") == 120


def test_parse_split_ratio_normalized():
    train, val, test = parse_split_ratio("60,20,20")
    assert abs(train - 0.6) < 1e-9
    assert abs(val - 0.2) < 1e-9
    assert abs(test - 0.2) < 1e-9


def test_assign_temporal_splits_is_chronological():
    windows = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:05:00",
            "2024-01-01 00:10:00",
            "2024-01-01 00:15:00",
            "2024-01-01 00:20:00",
        ]
    ).tolist()
    split_map = assign_temporal_splits(windows, (0.6, 0.2, 0.2))
    assert split_map[windows[0]] == "train"
    assert split_map[windows[1]] == "train"
    assert split_map[windows[2]] == "train"
    assert split_map[windows[3]] == "val"
    assert split_map[windows[4]] == "test"


def test_binary_label_mapping():
    s = pd.Series(["BENIGN", "Malicious", "normal", "attack", "0", "1"])
    out = parse_binary_label(s).tolist()
    assert out == [0, 1, 0, 1, 0, 1]


def test_pipeline_creates_expected_artifacts(tmp_path: Path):
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Timestamp,Source IP,Destination IP,Protocol,Flow Duration,Total Fwd Packets,Total Backward Packets,Total Length of Fwd Packets,Total Length of Bwd Packets,Flow Bytes/s,Flow Packets/s,Average Packet Size,Label",
                "2024-01-01 00:00:01,10.0.0.1,10.0.0.2,6,10,1,1,100,120,22,0.2,110,BENIGN",
                "2024-01-01 00:01:01,10.0.0.2,10.0.0.3,17,8,2,1,200,50,31.25,0.375,83.3,Malicious",
                "2024-01-01 00:02:01,10.0.0.3,10.0.0.1,6,9,2,2,150,180,36.67,0.444,82.5,BENIGN",
                "2024-01-01 00:08:01,10.0.0.1,10.0.0.3,6,7,1,1,90,80,24.28,0.285,85,Malicious",
            ]
        ),
        encoding="utf-8",
    )

    class Args:
        input_files = [str(csv_path)]
        input_glob = "*.csv"
        output_dir = str(tmp_path / "out")
        window_size = 5
        window_unit = "minutes"
        window = None
        node_label_rule = "any_malicious_wins"
        split_ratio = "0.6,0.2,0.2"
        max_rows_per_file = None
        seed = 42
        save_cleaned_flows = True
        log_level = "INFO"

    manifest = run_pipeline(Args())
    out_dir = Path(Args.output_dir)

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "schema.json").exists()
    assert (out_dir / "node_mapping.json").exists()
    assert (out_dir / "scaler_node.pkl").exists()
    assert (out_dir / "merged_cleaned_flows.csv").exists()
    assert manifest["scaling"]["fit_on"] == "train_only"
    assert manifest["summary"]["num_windows_total"] >= 1
    assert (out_dir / "graphs" / "train").exists()

