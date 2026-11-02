#!/usr/bin/env python3
"""Expose frozen NumPy indices to MATLAB and inventory metadata sources."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/cspp_training_distribution_20260909"
CHECKPOINT = Path("workspace/checkpoints/blind_v27c_C0_none_seed20261101")


def artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    stored = json.loads((CHECKPOINT / "protocol.json").read_text())
    upstream_path = ROOT / "experiments/domain_robust_v10_10_cwa2020_packet_context_materialization_protocol_20260719.json"
    upstream = json.loads(upstream_path.read_text())
    files = {
        "selected_indices": CHECKPOINT / "selected_indices.npz",
        "training_protocol": CHECKPOINT / "protocol.json",
        "instancegm_metadata": Path.home() / ".seisbench/datasets/instancegm/metadata.csv",
        "knet_metadata": ROOT / "data/knet_accel_causal_raw/metadata.csv",
        "knet_manifest": ROOT / "data/knet_accel_causal_raw/manifest.json",
        "cwa_store_metadata": Path(stored["cwa2020_opened_event_context_training"]["metadata"]),
        "cwa_source_metadata": Path(upstream["cwa_dataset_metadata"]["path"]),
        "cwa_materialization_protocol": upstream_path,
    }
    sources = {}
    for name, path in files.items():
        sources[name] = artifact(path)
        print(f"inventoried {name}", flush=True)
    assert sources["cwa_store_metadata"]["sha256"] == stored["cwa2020_opened_event_context_training"]["metadata_sha256"]
    assert sources["cwa_source_metadata"]["sha256"] == upstream["cwa_dataset_metadata"]["sha256"]
    indices = np.load(files["selected_indices"])
    selected = {name: indices[name] for name in ("ig_train", "knet_train", "cwa2020_opened_train")}
    assert [len(v) for v in selected.values()] == [1000, 1000, 500]
    assert np.array_equal(selected["ig_train"], indices["event_scope_train_instancegm"])
    assert np.array_equal(selected["knet_train"], indices["event_scope_train_knet"])
    savemat(OUTPUT / "selected_indices.mat", selected)
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(), "sources": sources,
        "indices": artifact(OUTPUT / "selected_indices.mat"),
        "selection": "Use exact final C0 checkpoint training indices; no new selection or model scoring",
        "chart_contract": {
            "question": "Where are the selected training earthquakes, and how are the selected waveform records distributed in magnitude and source depth?",
            "layout": "For each of InstanceGM, K-NET and CWA: one relief epicenter map and two histograms; no magnitude subgrouping",
            "map_grain": "Unique source event identifiers within each provider; metadata conflicts must be reported",
            "histogram_grain": "Selected waveform record; magnitude and depth repeated across stations reflect training record weighting",
            "histogram_rules": "Finite source metadata only; report every missing value; full numeric range; count axes begin at zero; arithmetic mean dashed line",
            "renderer": "MATLAB data extraction, aggregation and binning; GMT/PyGMT static PDF and PNG",
            "palette": "Reference-style cyan magnitude bars, green depth bars, red dashed means and red epicenters; real relief and blue ocean",
            "surface": "Standalone scientific figures; one combined figure and three source figures; no artificial fourth dataset",
        },
        "data_boundary": "Selected positive-event training records only; background and reviewed transient windows do not receive invented source locations or magnitudes",
        "code": [artifact(Path(__file__)), artifact(Path(__file__).with_name("aggregate_training_distribution.m"))],
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False)+"\n")
    print("Frozen training indices and metadata inventory ready for MATLAB", flush=True)


if __name__ == "__main__":
    main()
