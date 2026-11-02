#!/usr/bin/env python3
"""Freeze, replay, and analyze causal public baselines on frozen TexNet records."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import analyze_v27_ncedc_causal_public_baselines as public_analysis  # noqa: E402
import analyze_v27_ncedc_event_scores as event_analysis  # noqa: E402
import eval_v27_ncedc_causal_public_baselines as replay  # noqa: E402
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402


TX_ROOT = Path(
    "workspace/"
    "tx_external_event_validation_20260824"
)
TX_PROTOCOL = TX_ROOT / "protocol.json"
TX_SUMMARY = TX_ROOT / "results/summary.json"
TX_STATION_RESULTS = TX_ROOT / "results/event_station_results.csv"
SOURCE_PUBLIC_PROTOCOL = (
    ROOT / "experiments/cspp_v27_ncedc_causal_public_baseline_protocol_20260817.json"
)
PROTOCOL = TX_ROOT / "causal_public_baseline_protocol.json"
COHORT = TX_ROOT / "metadata/manifests/causal_public_baseline_cohort.csv"
ACQUISITIONS = TX_ROOT / "metadata/manifests/causal_public_baseline_waveforms.csv"
BASELINE_ROOT = TX_ROOT / "evaluation/tx_causal_public_baselines_v1"
ANALYSIS_ROOT = TX_ROOT / "evaluation/tx_causal_public_analysis_v1"
BASE_WRITE_JSON = replay.write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def csv_bytes(rows: Iterable[dict[str, Any]], fields: list[str]) -> bytes:
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(
        handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def station_key(row: dict[str, str]) -> str:
    return ".".join(
        (row["network"], row["station"], row["location"] or "--")
    )


def freeze() -> None:
    if any(path.exists() for path in (PROTOCOL, COHORT, ACQUISITIONS, BASELINE_ROOT)):
        raise FileExistsError("TexNet public-baseline protocol or output already exists")

    tx_protocol = json.loads(TX_PROTOCOL.read_text(encoding="utf-8"))
    tx_summary = json.loads(TX_SUMMARY.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_PUBLIC_PROTOCOL.read_text(encoding="utf-8"))
    requests = read_csv(Path(tx_protocol["inputs"]["event_station_requests"]["path"]))
    failed = {
        (row["event_id"], row["station_key"])
        for row in tx_summary["unscorable_event_station_records"]
    }
    cohort_rows = [
        {
            "event_id": row["event_id"],
            "station_key": station_key(row),
            "station_rank": int(row["station_rank"]),
        }
        for row in requests
        if (row["event_id"], station_key(row)) not in failed
    ]
    cohort_rows.sort(key=lambda row: (row["event_id"], row["station_rank"]))
    if len(cohort_rows) != 633 or len({row["event_id"] for row in cohort_rows}) != 65:
        raise RuntimeError("Unexpected TexNet same-record cohort")

    acquisition_rows = []
    for event_id in sorted(tx_summary["waveforms"]):
        spec = tx_summary["waveforms"][event_id]
        verify(spec)
        acquisition_rows.append(
            {
                "event_id": event_id,
                "mseed_file": spec["path"],
                "mseed_bytes": int(spec["bytes"]),
                "mseed_sha256": spec["sha256"],
            }
        )
    if len(acquisition_rows) != 65:
        raise RuntimeError("Unexpected TexNet waveform count")

    write_once(
        COHORT,
        csv_bytes(cohort_rows, ["event_id", "station_key", "station_rank"]),
    )
    write_once(
        ACQUISITIONS,
        csv_bytes(
            acquisition_rows,
            ["event_id", "mseed_file", "mseed_bytes", "mseed_sha256"],
        ),
    )

    code_paths = [
        "scripts/evaluation/eval_v27_tx_causal_public_baselines.py",
        "scripts/evaluation/eval_v27_ncedc_causal_public_baselines.py",
        "scripts/evaluation/analyze_v27_ncedc_causal_public_baselines.py",
        "scripts/evaluation/analyze_v27_ncedc_event_scores.py",
        "scripts/evaluation/build_v27_ncedc_event_score_cache.py",
        "scripts/evaluation/eval_v27c_fixed_public_baselines.py",
        "src/oriented_gappy_strong_motion_stream.py",
        "src/resilient_mseed.py",
        "src/stream_quality_v5.py",
        "src/v20c_public_window_baselines.py",
    ]
    event_ids = sorted(
        {row["event_id"] for row in cohort_rows},
        key=lambda event_id: next(
            row["origin_time_utc"] for row in requests if row["event_id"] == event_id
        ),
    )
    payload = {
        "protocol_id": "cspp_v27_tx_causal_public_baseline_v1",
        "status": "frozen_before_any_texnet_public_baseline_output",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "texnet_protocol": artifact(TX_PROTOCOL),
            "texnet_summary": artifact(TX_SUMMARY),
            "texnet_station_results": artifact(TX_STATION_RESULTS),
            "event_station_requests": artifact(
                Path(tx_protocol["inputs"]["event_station_requests"]["path"])
            ),
            "event_station_orientations": artifact(
                Path(tx_protocol["inputs"]["event_station_orientations"]["path"])
            ),
            "cohort": artifact(COHORT),
            "acquisition_status": artifact(ACQUISITIONS),
            "source_public_model_protocol": artifact(SOURCE_PUBLIC_PROTOCOL),
        },
        "cohort": {
            "events": 65,
            "records": 633,
            "event_ids": event_ids,
            "selection": (
                "All deterministic data-bearing records in the frozen TexNet "
                "event cohort, independent of CSPP detection success and all "
                "public-model outputs."
            ),
        },
        "models": source["models"],
        "causal_visibility": {
            "record_duration_seconds": 240.0,
            "pre_origin_history_seconds": 60.0,
            "packet_clock_seconds": 0.5,
            "future_samples_after_publication_time": 0,
            "native_windows_and_blinding_retained": True,
            "publication": (
                "Each annotation sample is published from the earliest fully "
                "received native window; a pick becomes available only after "
                "its threshold episode closes."
            ),
        },
        "statistics": {
            "bootstrap_unit": "event",
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 20260824,
            "match_tolerance_seconds": 1.5,
            "snr_bins_db": ["lt_0", "0_to_lt_10", "ge_10", "invalid"],
            "snr_role": "posthoc_diagnostic_only_not_an_online_input",
        },
        "execution": {
            "device": "mps",
            "workers": 1,
            "serial_models": True,
            "maximum_process_rss_bytes": 4 * 1024**3,
            "global_mps_lock": str(TX_ROOT / "evaluation/.public_mps.lock"),
            "output_root": str(BASELINE_ROOT),
            "analysis_output_root": str(ANALYSIS_ROOT),
        },
        "code_sha256": {
            relative: sha256_file(ROOT / relative) for relative in code_paths
        },
        "prohibited": [
            "threshold selection on TexNet",
            "record removal based on CSPP or public-model success",
            "future samples after the declared publication time",
            "parallel MPS model processes",
            "automatic manuscript modification",
        ],
    }
    write_once(PROTOCOL, json_bytes(payload))
    print(json.dumps(artifact(PROTOCOL), indent=2))


def load_protocol(path: Path, expected: str) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"TexNet public-baseline protocol SHA-256 differs: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_tx_causal_public_baseline_v1":
        raise RuntimeError("Unexpected TexNet public-baseline protocol")
    for relative, digest in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != digest:
            raise RuntimeError(f"Public-baseline code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify(spec)
    for model in protocol["models"].values():
        for spec in model["artifacts"].values():
            verify(spec)
    return protocol, actual


def load_cohort_keys(protocol: dict[str, Any]) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(verify(protocol["inputs"]["cohort"])):
        keys[row["event_id"]].add(row["station_key"].replace(".--", "."))
    if sum(map(len, keys.values())) != int(protocol["cohort"]["records"]):
        raise RuntimeError("TexNet public-baseline cohort count differs")
    return keys


def tx_write_json(path: Path, payload: Any) -> None:
    if isinstance(payload, dict) and payload.get("summary_id") == (
        "cspp_v27_ncedc_causal_public_event_v1"
    ):
        payload = dict(payload)
        payload["summary_id"] = "cspp_v27_tx_causal_public_event_v1"
    BASE_WRITE_JSON(path, payload)


def finalize(
    protocol: dict[str, Any], protocol_sha256: str, model_key: str
) -> None:
    root = Path(protocol["execution"]["output_root"]) / model_key
    summaries = []
    rows = 0
    for event_id in protocol["cohort"]["event_ids"]:
        path = root / "events" / event_id / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256") != protocol_sha256
            or summary.get("model_key") != model_key
            or summary.get("summary_id") != "cspp_v27_tx_causal_public_event_v1"
        ):
            raise RuntimeError(f"Public-baseline event summary differs: {path}")
        verify(summary["records"])
        summaries.append(artifact(path))
        rows += int(summary["row_count"])
    completion = {
        "completion_id": f"cspp_v27_tx_causal_public_{model_key}_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "model_key": model_key,
        "events": len(summaries),
        "records": int(protocol["cohort"]["records"]),
        "rows": rows,
        "event_summaries": summaries,
    }
    BASE_WRITE_JSON(root / "completion.json", completion)
    print(json.dumps(completion | {"event_summaries": "omitted"}, indent=2))


def run_replay() -> None:
    replay.DEFAULT_PROTOCOL = PROTOCOL
    replay.load_protocol = load_protocol
    replay.load_cohort_keys = load_cohort_keys
    replay.write_json = tx_write_json
    replay.finalize = finalize
    replay.main()


def snr_level(row: dict[str, str]) -> str:
    if row["snr_valid"].lower() != "true" or not row["snr_db"]:
        return "invalid"
    value = float(row["snr_db"])
    if value < 0.0:
        return "lt_0"
    if value < 10.0:
        return "0_to_lt_10"
    return "ge_10"


def baseline_rows(protocol: dict[str, Any], protocol_sha: str) -> list[dict[str, str]]:
    rows = public_analysis.read_rows(protocol, protocol_sha)
    expected = int(protocol["cohort"]["records"])
    for model_key, model in protocol["models"].items():
        count = sum(row["model_key"] == model_key for row in rows)
        if count != expected * len(model["operating_points"]):
            raise RuntimeError(f"Incomplete public-baseline matrix: {model_key}")
    return rows


def normalized_cspp_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tx_summary = json.loads(verify(protocol["inputs"]["texnet_summary"]).read_text())
    primary = tx_summary["primary_model_key"]
    result = []
    for row in read_csv(verify(protocol["inputs"]["texnet_station_results"])):
        if row["model_key"] != primary:
            continue
        value: dict[str, Any] = dict(row)
        value["matched"] = row["matched"].lower() == "true"
        value["matched_candidate_count"] = int(row["matched_candidate_count"])
        value["candidate_count"] = int(row["candidate_count"])
        value["snr_level"] = snr_level(row)
        result.append(value)
    return result


def low_snr_diagnosis(
    protocol: dict[str, Any], cspp_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tx_protocol = json.loads(verify(protocol["inputs"]["texnet_protocol"]).read_text())
    candidate_threshold = float(tx_protocol["operating_point"]["candidate_threshold"])
    event_threshold = float(tx_protocol["operating_point"]["event_threshold"])
    history_packets = int(tx_protocol["runtime"]["candidate_history_packets"])
    packet_seconds = float(tx_protocol["runtime"]["packet_duration_seconds"])
    primary_key = json.loads(
        verify(protocol["inputs"]["texnet_summary"]).read_text()
    )["primary_model_key"]
    requests = {
        (row["event_id"], station_key(row)): row
        for row in read_csv(verify(protocol["inputs"]["event_station_requests"]))
    }
    details = []
    for row in cspp_rows:
        if row["snr_level"] != "lt_0":
            continue
        request = requests[(row["event_id"], row["station_key"])]
        reference = float(replay.UTCDateTime(request["primary_pick_time_utc"]).timestamp)
        start = float(replay.UTCDateTime(request["request_start_utc"]).timestamp)
        archive = np.load(row["score_cache"])
        candidate = archive["candidate_probability"].astype(np.float64)
        event = archive[f"event_probability__{primary_key}"].astype(np.float64)
        valid = archive["quality_valid"].astype(bool)
        allowed = archive["candidate_allowed"].astype(bool)
        support = event_analysis.support_with_resets(
            candidate, allowed, valid, candidate_threshold, history_packets
        )
        p_packet = int(math.floor((reference - start) / packet_seconds))
        first = max(0, p_packet - 4)
        last = min(len(candidate), p_packet + 5)
        window = slice(first, last)
        candidate_hit = bool(
            np.any(valid[window] & allowed[window] & (candidate[window] >= candidate_threshold))
        )
        support_hit = bool(np.any(valid[window] & support[window]))
        event_hit = bool(np.any(valid[window] & (event[window] >= event_threshold)))
        joint_hit = bool(
            np.any(valid[window] & support[window] & (event[window] >= event_threshold))
        )
        if row["matched"]:
            failure_stage = "matched"
        elif not support_hit:
            failure_stage = "candidate_evidence_below_threshold"
        elif not event_hit:
            failure_stage = "event_confirmation_below_threshold"
        elif not joint_hit:
            failure_stage = "gates_not_simultaneous"
        else:
            failure_stage = "timing_or_episode_closure_mismatch"
        details.append(
            {
                "event_id": row["event_id"],
                "station_key": row["station_key"],
                "snr_db": float(row["snr_db"]),
                "snr_left_censored": float(row["snr_db"]) <= -120.0,
                "matched": bool(row["matched"]),
                "candidate_count": int(row["candidate_count"]),
                "candidate_gate_hit_near_reference": candidate_hit,
                "candidate_support_hit_near_reference": support_hit,
                "event_gate_hit_near_reference": event_hit,
                "joint_gate_hit_near_reference": joint_hit,
                "candidate_probability_max_near_reference": float(
                    np.nanmax(candidate[window])
                ),
                "event_probability_max_near_reference": float(np.nanmax(event[window])),
                "failure_stage": failure_stage,
            }
        )
    return details, {
        "records": len(details),
        "matched_records": sum(row["matched"] for row in details),
        "left_censored_at_minus_120_db": sum(
            row["snr_left_censored"] for row in details
        ),
        "station_counts": dict(Counter(row["station_key"] for row in details)),
        "failure_stage_counts": dict(Counter(row["failure_stage"] for row in details)),
        "diagnostic_window": "four 0.5-s packets before through four after the reference packet",
        "interpretation": (
            "The reference-pick SNR uses post-arrival samples and is diagnostic; "
            "it is unavailable to a live trigger and cannot serve as an online gate."
        ),
    }


def analyze(expected: str) -> None:
    protocol, protocol_sha = load_protocol(PROTOCOL, expected)
    if ANALYSIS_ROOT.exists():
        raise FileExistsError(ANALYSIS_ROOT)
    rows = baseline_rows(protocol, protocol_sha)
    cspp_rows = normalized_cspp_rows(protocol)
    snr_by_key = {
        (row["event_id"], row["station_key"]): row["snr_level"]
        for row in cspp_rows
    }

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["method_id"])].append(row)
    results = {}
    strata = {}
    for (model_key, method_id), values in sorted(grouped.items()):
        key = f"{model_key}:{method_id}"
        result = public_analysis.metrics(values)
        result["event_cluster_bootstrap95"] = public_analysis.bootstrap(
            values,
            int(protocol["statistics"]["bootstrap_replicates"]),
            int(protocol["statistics"]["bootstrap_seed"]),
        )
        results[key] = result
        strata[key] = {}
        for level in ("lt_0", "0_to_lt_10", "ge_10", "invalid"):
            subset = [
                row
                for row in values
                if snr_by_key[(row["event_id"], row["station_key"])] == level
            ]
            strata[key][level] = public_analysis.metrics(subset)

    cspp_strata = {
        level: event_analysis.metrics(
            [row for row in cspp_rows if row["snr_level"] == level]
        )
        for level in ("lt_0", "0_to_lt_10", "ge_10", "invalid")
    }
    details, diagnosis = low_snr_diagnosis(protocol, cspp_rows)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=False)
    detail_path = ANALYSIS_ROOT / "cspp_low_snr_diagnosis.csv"
    write_once(detail_path, csv_bytes(details, list(details[0])))
    payload = {
        "analysis_id": "cspp_v27_tx_causal_public_analysis_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha,
        "same_record_causal_public_baselines": results,
        "snr_stratified_public_baselines": strata,
        "snr_stratified_cspp": cspp_strata,
        "cspp_low_snr_diagnosis": diagnosis,
        "artifacts": {"low_snr_record_diagnosis": artifact(detail_path)},
        "interpretation": {
            "scope": (
                "All methods use the same 633 TexNet records and strict causal "
                "publication; public checkpoints retain native preprocessing, "
                "windowing, blinding, and previously fixed thresholds."
            ),
            "training": "released-model comparison, not matched retraining",
            "snr": (
                "SNR strata are posthoc diagnostics and were not used for record "
                "selection, threshold tuning, or online decisions."
            ),
            "paper_or_submission_files_modified": False,
        },
    }
    result_path = ANALYSIS_ROOT / "result.json"
    write_once(result_path, json_bytes(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def selftest() -> None:
    assert snr_level({"snr_valid": "False", "snr_db": ""}) == "invalid"
    assert snr_level({"snr_valid": "True", "snr_db": "-0.1"}) == "lt_0"
    assert snr_level({"snr_valid": "True", "snr_db": "9.9"}) == "0_to_lt_10"
    assert snr_level({"snr_valid": "True", "snr_db": "10"}) == "ge_10"
    print("selftest passed")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: script.py freeze|run|analyze|selftest [options]")
    stage = sys.argv.pop(1)
    if stage == "freeze":
        freeze()
    elif stage == "run":
        run_replay()
    elif stage == "analyze":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--expected-protocol-sha256", required=True)
        args = parser.parse_args()
        analyze(args.expected_protocol_sha256)
    elif stage == "selftest":
        selftest()
    else:
        raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    main()
