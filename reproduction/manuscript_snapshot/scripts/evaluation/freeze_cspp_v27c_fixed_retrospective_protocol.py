#!/usr/bin/env python3
"""Freeze the V27C retrospective K-NET/InstanceGM benchmark protocol."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = Path(
    "workspace"
).resolve()
DEFAULT_SELECTION_FREEZE = (
    ROOT
    / "experiments/cspp_v27_geonet_temporal_selection_freeze_20260729.json"
)
DEFAULT_V24_PROTOCOL = (
    ROOT
    / "experiments/v24_station_level_external_test_20260728/"
    "station_level_test_protocol.json"
)
DEFAULT_OUTPUT = (
    ALLOWED_OUTPUT_ROOT
    / "protocol_freeze/cspp_v27c_fixed_retrospective_protocol_20260729.json"
)
DEFAULT_RUN_ROOT = (
    ALLOWED_OUTPUT_ROOT
    / "evaluation/cspp_v27c_fixed_retrospective_20260729/runs"
)
DEFAULT_ANALYSIS_ROOT = (
    ALLOWED_OUTPUT_ROOT
    / "evaluation/cspp_v27c_fixed_retrospective_20260729/analysis"
)
EXPECTED_COUNTS = {"knet": 701, "instancegm": 1170}
SELECTION_FREEZE_ID = "cspp_v27_geonet_temporal_selection_freeze_v1"
PROTOCOL_ID = "cspp_v27c_fixed_retrospective_benchmark_v1"
DEVELOPMENT_PROTOCOL_ID = (
    "cspp_v27_geonet_temporal_evaluation_execution_v3"
)
EXPECTED_DEVELOPMENT_PROTOCOL_SHA256 = (
    "2b98f6833b90bc62aa44c647be5f16845b376ad7ebee7a3d308d13e67f947047"
)
EXPECTED_V24_PROTOCOL_SHA256 = (
    "e7f6d1a4e4b3d9c3225b1f035f964149e903925d44de454eca8bfa1d1a96a9e9"
)
EXPECTED_TRAINING_SELECTION_REFERENCE_SHA256 = (
    "72dab23b14f4da309ec04bbbcdb68c7088bbf8aba565024934c2be05c94893cd"
)
DEFAULT_WAVEFORM_HASH_REFERENCES = {
    "knet": Path(
        "external_volume/CSPP_current/dependencies/simple_v25_work_20260728/evaluation/"
        "cspp_v25_fixed_test_20260728/runs/knet/records.csv"
    ),
    "instancegm": Path(
        "external_volume/CSPP_current/dependencies/simple_v25_work_20260728/evaluation/"
        "cspp_v25_fixed_test_20260728/runs/instancegm/records.csv"
    ),
}
EXPECTED_WAVEFORM_HASH_REFERENCE_SHA256 = {
    "knet": (
        "89d9669c722987066d8b8a78b259826f534cf5261b261a706afe0d2618d73af4"
    ),
    "instancegm": (
        "7c7fc9b1033906e037d74e537b677987a8049692e3638adceafdfa2713a2a56b"
    ),
}
FORBIDDEN_VOLUME = Path("external_volume")

CODE_PATHS = (
    "scripts/evaluation/freeze_cspp_v27c_fixed_retrospective_protocol.py",
    "scripts/evaluation/eval_cspp_v27c_fixed_retrospective.py",
    "scripts/evaluation/eval_v24_fixed_test_models.py",
    "scripts/evaluation/eval_v27c_fixed_public_baselines.py",
    "scripts/evaluation/freeze_v27c_fixed_public_baseline_protocol.py",
    "scripts/evaluation/run_cspp_v27c_fixed_retrospective.py",
    "scripts/evaluation/run_v20c_geonet_score_cache_safe.py",
    "scripts/evaluation/run_v27c_fixed_public_baselines.py",
    "scripts/evaluation/build_v20c_geonet_score_cache.py",
    "scripts/evaluation/v27_runner_lease.py",
    "scripts/evaluation/analyze_cspp_v27c_fixed_retrospective.py",
    "scripts/evaluation/analyze_v27_geonet_temporal_selection.py",
    "src/causal_fir_decimator.py",
    "src/causal_packet_veto_features.py",
    "src/causal_stream_veto_features.py",
    "src/config.py",
    "src/cspp_v27_reused_core_score_cache.py",
    "src/current_packet_artifact_features.py",
    "src/current_packet_artifact_head.py",
    "src/data_streaming.py",
    "src/domain_robust_model_io.py",
    "src/gappy_strong_motion_stream.py",
    "src/gappy_strong_motion_stream_v2.py",
    "src/model_causal_classical_v1.py",
    "src/model_cspp_v25.py",
    "src/model_cspp_v26.py",
    "src/model_cspp_v27.py",
    "src/model_domain_robust_v10.py",
    "src/model_domain_robust_v11.py",
    "src/model_domain_robust_v5.py",
    "src/model_domain_robust_v6.py",
    "src/model_domain_robust_v7.py",
    "src/model_domain_robust_v8.py",
    "src/model_domain_robust_v9.py",
    "src/model_packet_once_v1.py",
    "src/oriented_gappy_strong_motion_stream.py",
    "src/resilient_mseed.py",
    "src/stream_quality_v5.py",
    "src/v15_reused_packet_features.py",
    "src/v20c_causal_score_cache.py",
    "src/v20c_event_snr.py",
    "src/v20c_public_window_baselines.py",
    "src/v21_causal_score_cache.py",
    "src/v21_score_shape_calibrator.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-freeze",
        type=Path,
        default=DEFAULT_SELECTION_FREEZE,
    )
    parser.add_argument(
        "--expected-selection-sha256",
        required=True,
        help="Required SHA-256 of the independently sealed GeoNet selection",
    )
    parser.add_argument(
        "--v24-protocol", type=Path, default=DEFAULT_V24_PROTOCOL
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT
    )
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--expected-parent-protocol-sha256")
    parser.add_argument("--failed-attempt-root", type=Path)
    args = parser.parse_args()
    amendment_values = (
        args.parent_protocol,
        args.expected_parent_protocol_sha256,
        args.failed_attempt_root,
    )
    if any(value is not None for value in amendment_values) and not all(
        value is not None for value in amendment_values
    ):
        parser.error(
            "--parent-protocol, --expected-parent-protocol-sha256, and "
            "--failed-attempt-root must be supplied together"
        )
    return args


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reject_forbidden_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if is_within(resolved, FORBIDDEN_VOLUME):
        raise RuntimeError(f"WD_BLACK is forbidden for this benchmark: {resolved}")
    return resolved


def require_output_path(
    path: str | Path,
    *,
    allowed_root: Path = ALLOWED_OUTPUT_ROOT,
) -> Path:
    resolved = reject_forbidden_path(path)
    if not is_within(resolved, allowed_root):
        raise RuntimeError(
            f"Benchmark output must remain under {allowed_root}: {resolved}"
        )
    return resolved


def artifact(path: str | Path, *, hash_file: bool = True) -> dict[str, Any]:
    resolved = reject_forbidden_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved) if hash_file else None,
    }


def verify_artifact(
    spec: Mapping[str, Any],
    *,
    require_sha256: bool,
    require_mtime: bool = False,
) -> Path:
    path = reject_forbidden_path(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if int(spec["bytes"]) != int(stat.st_size):
        raise RuntimeError(f"Artifact byte count differs: {path}")
    if require_mtime and int(spec["mtime_ns"]) != int(stat.st_mtime_ns):
        raise RuntimeError(f"Artifact mtime differs: {path}")
    expected = spec.get("sha256")
    if require_sha256 and not expected:
        raise RuntimeError(f"Artifact has no frozen SHA-256: {path}")
    if expected and sha256_file(path) != str(expected):
        raise RuntimeError(f"Artifact SHA-256 differs: {path}")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = reject_forbidden_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {resolved}")
    return payload


def execution_amendment(
    *,
    parent_protocol_path: Path,
    expected_parent_protocol_sha256: str,
    failed_attempt_root: Path,
    allowed_output_root: Path = ALLOWED_OUTPUT_ROOT,
) -> dict[str, Any]:
    parent_path = reject_forbidden_path(parent_protocol_path)
    if (
        not parent_path.is_file()
        or sha256_file(parent_path)
        != expected_parent_protocol_sha256
    ):
        raise RuntimeError("Parent retrospective protocol identity differs")
    parent = load_json(parent_path)
    if parent.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected parent retrospective protocol")

    attempt = require_output_path(
        failed_attempt_root, allowed_root=allowed_output_root
    )
    if not attempt.is_dir():
        raise RuntimeError("Failed retrospective attempt is absent")
    required = {
        name: attempt / name
        for name in ("runner_lease.json", "summary.json", "records.csv")
    }
    if any(not path.is_file() for path in required.values()):
        raise RuntimeError("Failed retrospective attempt is incomplete")
    if (attempt / "execution_audit.json").exists():
        raise RuntimeError(
            "The supplied attempt already has an execution audit"
        )
    summary = load_json(required["summary.json"])
    lease = load_json(required["runner_lease.json"])
    parent_runner_sha256 = parent["code_sha256"][
        "scripts/evaluation/run_cspp_v27c_fixed_retrospective.py"
    ]
    parent_evaluator_sha256 = parent["code_sha256"][
        "scripts/evaluation/eval_cspp_v27c_fixed_retrospective.py"
    ]
    expected_published = (
        Path(parent["execution"]["score_output_root"]) / "knet"
    ).resolve()
    if (
        summary.get("summary_id")
        != "cspp_v27c_fixed_retrospective_score_v1"
        or summary.get("protocol_sha256")
        != expected_parent_protocol_sha256
        or summary.get("dataset") != "knet"
        or int(summary.get("records_requested", -1))
        != EXPECTED_COUNTS["knet"]
        or int(summary.get("records_emitted", -1))
        != EXPECTED_COUNTS["knet"]
        or lease.get("protocol_sha256")
        != expected_parent_protocol_sha256
        or lease.get("unit") != {"dataset": "knet"}
        or lease.get("runner_script_sha256")
        != parent_runner_sha256
        or lease.get("evaluator_script_sha256")
        != parent_evaluator_sha256
        or Path(str(lease.get("attempt_root", ""))).resolve() != attempt
        or Path(str(lease.get("published_root", ""))).resolve()
        != expected_published
        or expected_published.exists()
    ):
        raise RuntimeError("Failed retrospective attempt identity differs")
    current_runner_sha256 = sha256_file(
        ROOT
        / "scripts/evaluation/run_cspp_v27c_fixed_retrospective.py"
    )
    if current_runner_sha256 == parent_runner_sha256:
        raise RuntimeError("Execution amendment does not change the runner")
    return {
        "amendment_id": (
            "cspp_v27c_fixed_retrospective_execution_amendment_a1"
        ),
        "parent_protocol": artifact(parent_path),
        "failed_attempt": {
            "root": str(attempt),
            **{
                name.removesuffix(".json").removesuffix(".csv"): artifact(
                    path
                )
                for name, path in required.items()
            },
            "execution_audit_absent": True,
            "published_root_absent": True,
        },
        "observed_failure": (
            "K-NET emitted all 701 records and exited with return code zero, "
            "then the parent sampled the already-exited PID and classified "
            "the resulting missing RSS sample as a live resource failure."
        ),
        "code_change": (
            "Poll before sampling and accept an RSS-disappearance result "
            "only when a second poll proves the child exited during that "
            "sample. A live child with an unreadable process tree still "
            "fails closed."
        ),
        "parent_runner_sha256": parent_runner_sha256,
        "amended_runner_sha256": current_runner_sha256,
        "scientific_configuration_changed": False,
        "previous_attempt_reused": False,
        "rerun_required": True,
        "claim_boundary": (
            "A1 changes only process-lifecycle handling after a completed "
            "child exits. It cannot change records, labels, model weights, "
            "thresholds, preprocessing, metrics, or dataset membership."
        ),
    }


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _same_artifact_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return (
        reject_forbidden_path(str(left["path"]))
        == reject_forbidden_path(str(right["path"]))
        and int(left["bytes"]) == int(right["bytes"])
        and str(left["sha256"]) == str(right["sha256"])
    )


def resolve_selected_model(
    selection_freeze: Mapping[str, Any],
    *,
    expected_development_protocol_sha256: str = (
        EXPECTED_DEVELOPMENT_PROTOCOL_SHA256
    ),
    expected_training_selection_reference_sha256: str = (
        EXPECTED_TRAINING_SELECTION_REFERENCE_SHA256
    ),
) -> dict[str, Any]:
    if selection_freeze.get("status") != (
        "development_arm_threshold_and_primary_seed_irrevocably_frozen"
    ):
        raise RuntimeError("Development selection is not irrevocably frozen")
    point = selection_freeze.get("operating_point")
    if not isinstance(point, dict):
        raise RuntimeError("Selection seal has no operating point")
    arm = str(point.get("selected_arm", ""))
    if arm not in {"C0", "C1"}:
        raise RuntimeError(f"Unexpected selected arm: {arm}")
    seed = int(
        point.get(
            "primary_operational_seed",
            point.get("primary_seed", -1),
        )
    )
    candidate_threshold = float(point.get("candidate_threshold", math.nan))
    event_threshold = float(point.get("event_threshold", math.nan))
    if not math.isclose(candidate_threshold, 0.1, abs_tol=1e-12):
        raise RuntimeError("Retrospective candidate threshold must be 0.1")
    if not math.isfinite(event_threshold) or not 0.0 < event_threshold <= 1.0:
        raise RuntimeError("Selection freeze has an invalid event threshold")
    all_seeds = [int(value) for value in point.get("all_reported_seeds", [])]
    if (
        seed != 20261101
        or all_seeds != [20261101, 20261102, 20261103]
        or point.get("seed_selection_permitted") is not False
    ):
        raise RuntimeError("Selection seal seed policy differs")

    inputs = selection_freeze.get("selection_inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("Selection seal has no selection_inputs")
    result_spec = inputs.get("selection_result")
    protocol_spec = inputs.get("evaluation_protocol")
    if not isinstance(result_spec, dict) or not isinstance(
        protocol_spec, dict
    ):
        raise RuntimeError(
            "Selection seal does not bind result and evaluation protocol"
        )
    expected_unlock = stable_digest(
        {"operating_point": point, "selection_inputs": inputs}
    )
    if selection_freeze.get("unlock_token") != expected_unlock:
        raise RuntimeError("Selection-seal unlock token differs")

    result_path = verify_artifact(
        result_spec, require_sha256=True, require_mtime=False
    )
    protocol_path = verify_artifact(
        protocol_spec, require_sha256=True, require_mtime=False
    )
    if str(protocol_spec["sha256"]) != (
        expected_development_protocol_sha256
    ):
        raise RuntimeError(
            "Selection seal does not bind the frozen V3 evaluation protocol"
        )
    result = load_json(result_path)
    if (
        result.get("result_id")
        != "cspp_v27_geonet_temporal_selection_result_v1"
    ):
        raise RuntimeError("Unexpected GeoNet selection result")
    development_protocol = load_json(protocol_path)
    if development_protocol.get("protocol_id") != DEVELOPMENT_PROTOCOL_ID:
        raise RuntimeError("Unexpected GeoNet development protocol")
    result_protocol_spec = result.get("protocol")
    if not isinstance(result_protocol_spec, dict) or not _same_artifact_identity(
        result_protocol_spec, protocol_spec
    ):
        raise RuntimeError(
            "Selection result and seal bind different evaluation protocols"
        )
    result_point = result.get("frozen_operating_point")
    if not isinstance(result_point, dict):
        raise RuntimeError("Selection result has no frozen operating point")
    for key in (
        "selected_arm",
        "candidate_threshold",
        "event_threshold",
        "primary_operational_seed",
        "all_reported_seeds",
    ):
        if result_point.get(key) != point.get(key):
            raise RuntimeError(
                f"Selection result and seal operating points differ: {key}"
            )

    selection = development_protocol.get("selection", {})
    if int(selection.get("primary_seed", -1)) != seed:
        raise RuntimeError("Primary seed differs from development protocol")
    if not math.isclose(
        float(selection.get("candidate_threshold", math.nan)),
        candidate_threshold,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Candidate threshold differs from development protocol"
        )
    if event_threshold not in {
        float(value) for value in selection.get("event_threshold_grid", [])
    }:
        raise RuntimeError(
            "Selected event threshold was not in the frozen grid"
        )

    selected_models = selection_freeze.get("selected_models")
    if not isinstance(selected_models, dict):
        raise RuntimeError("Selection seal has no selected_models")
    if sorted(int(spec["seed"]) for spec in selected_models.values()) != all_seeds:
        raise RuntimeError("Selection seal checkpoint set is incomplete")
    if any(str(spec.get("arm")) != arm for spec in selected_models.values()):
        raise RuntimeError("Selection seal contains a model from another arm")

    primary_matches: list[dict[str, Any]] = []
    for key, sealed_spec in selected_models.items():
        if not isinstance(sealed_spec, dict):
            raise RuntimeError(f"Invalid sealed model entry: {key}")
        model_path = verify_artifact(
            sealed_spec, require_sha256=True, require_mtime=False
        )
        development_spec = development_protocol.get("models", {}).get(key)
        if not isinstance(development_spec, dict):
            raise RuntimeError(
                f"Sealed model is absent from evaluation protocol: {key}"
            )
        identity_fields = (
            "path",
            "bytes",
            "sha256",
            "arm",
            "seed",
            "verifier_feature_mode",
        )
        for field in identity_fields:
            if sealed_spec.get(field) != development_spec.get(field):
                raise RuntimeError(
                    f"Sealed checkpoint differs from protocol: {key}:{field}"
                )
        frozen_spec = {
            **dict(sealed_spec),
            **artifact(model_path),
        }
        if int(sealed_spec["seed"]) == seed:
            primary_matches.append(frozen_spec)
    if len(primary_matches) != 1:
        raise RuntimeError("Selection seal has no unique primary checkpoint")
    checkpoint_spec = primary_matches[0]
    checkpoint = torch.load(
        checkpoint_spec["path"],
        map_location="cpu",
        weights_only=False,
    )
    training_config = checkpoint.get("training_config", {})
    model_config = checkpoint.get("model_config", {})
    if (
        checkpoint.get("model_class") != "CSPPCausalPickerV27C"
        or int(checkpoint.get("epoch", -1)) != 3
        or int(training_config.get("seed", -1)) != seed
        or str(model_config.get("verifier_feature_mode"))
        != str(checkpoint_spec["verifier_feature_mode"])
    ):
        raise RuntimeError("Selected checkpoint internal metadata differs")
    selection_reference_path = reject_forbidden_path(
        training_config.get("selection_reference_npz", "")
    )
    if (
        not selection_reference_path.is_file()
        or sha256_file(selection_reference_path)
        != expected_training_selection_reference_sha256
    ):
        raise RuntimeError(
            "Selected checkpoint training-selection reference differs"
        )

    evidence = selection_freeze.get("evidence_boundary", {})
    if (
        evidence.get("arm_selected_once") is not True
        or evidence.get("threshold_selected_once") is not True
        or evidence.get("primary_seed_was_predeclared") is not True
        or evidence.get("final_holdout_root_absent_at_freeze") is not True
        or evidence.get(
            "final_holdout_waveforms_accessed_before_freeze"
        )
        is not False
    ):
        raise RuntimeError("Selection seal evidence boundary differs")

    return {
        "selected_arm": arm,
        "primary_seed": seed,
        "candidate_threshold": candidate_threshold,
        "event_threshold": event_threshold,
        "checkpoint": checkpoint_spec,
        "training_selection_reference": artifact(
            selection_reference_path
        ),
        "selection_result": artifact(result_path),
        "development_protocol": artifact(protocol_path),
    }


def read_manifest(
    path: Path,
    *,
    dataset: str,
    expected_count: int,
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "source_index",
        "trace_name",
        "event_id",
        "station_id",
        "split",
        "magnitude",
        "epicentral_distance_km",
        "p_arrival_sample",
        "sample_rate_hz",
        "n_samples",
    }
    if len(rows) != int(expected_count):
        raise RuntimeError(
            f"{dataset} manifest has {len(rows)} rows, expected "
            f"{expected_count}"
        )
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f"{dataset} manifest lacks columns: {sorted(missing)}")
    keys = {
        (row["source_index"], row["trace_name"], row["event_id"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise RuntimeError(f"{dataset} manifest has duplicate record keys")
    if any(
        not math.isclose(float(row["sample_rate_hz"]), 100.0, abs_tol=1e-9)
        for row in rows
    ):
        raise RuntimeError(f"{dataset} manifest is not uniformly 100 Hz")
    if any(int(float(row["n_samples"])) < 50 for row in rows):
        raise RuntimeError(f"{dataset} manifest contains an empty trace")
    if any(row["split"] != "test" for row in rows):
        raise RuntimeError(f"{dataset} manifest is not provider test")
    if any(float(row["magnitude"]) < 4.0 for row in rows):
        raise RuntimeError(f"{dataset} manifest contains M < 4")
    if any(
        not 0.0 <= float(row["epicentral_distance_km"]) <= 200.0
        for row in rows
    ):
        raise RuntimeError(
            f"{dataset} manifest contains a distance outside 200 km"
        )
    if any(
        not 0.0 <= float(row["p_arrival_sample"])
        < int(float(row["n_samples"]))
        for row in rows
    ):
        raise RuntimeError(
            f"{dataset} manifest contains an invalid P arrival"
        )
    if dataset == "knet":
        required_knet = {
            "component_order",
            "input_units",
            "preprocess_method",
        }
        missing_knet = required_knet.difference(rows[0])
        if missing_knet:
            raise RuntimeError(
                f"knet manifest lacks columns: {sorted(missing_knet)}"
            )
        component_orders = {row["component_order"] for row in rows}
        if component_orders != {"ZNE"}:
            raise RuntimeError(
                f"knet manifest component order differs: {component_orders}"
            )
        if {row["input_units"] for row in rows} != {
            "raw_acceleration_native_bson_values"
        }:
            raise RuntimeError("knet manifest input units differ")
        if {row["preprocess_method"] for row in rows} != {
            "raw-acceleration-no-offline-filter"
        }:
            raise RuntimeError("knet preprocessing differs")
    else:
        required_instance = {"station_channels", "input_units"}
        missing_instance = required_instance.difference(rows[0])
        if missing_instance:
            raise RuntimeError(
                "instancegm manifest lacks columns: "
                f"{sorted(missing_instance)}"
            )
        if not {
            row["station_channels"] for row in rows
        }.issubset({"HN", "EN"}):
            raise RuntimeError("InstanceGM channel family differs")
        if {row["input_units"] for row in rows} != {"mps2"}:
            raise RuntimeError("InstanceGM input units differ")
    return rows


def training_test_overlap_audit(
    selection_reference_path: Path,
    manifests: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    key_map = {
        "knet": (
            "knet_train",
            "knet_noise_train",
            "event_scope_train_knet",
        ),
        "instancegm": (
            "ig_train",
            "ig_noise_train",
            "event_scope_train_instancegm",
        ),
    }
    datasets = {}
    with np.load(selection_reference_path, allow_pickle=False) as selected:
        for dataset, keys in key_map.items():
            test_indices = {
                int(row["source_index"]) for row in manifests[dataset]
            }
            key_results = {}
            for key in keys:
                if key not in selected.files:
                    raise RuntimeError(
                        f"Training-selection reference lacks {key}"
                    )
                training_indices = {
                    int(value)
                    for value in np.asarray(selected[key]).reshape(-1)
                }
                overlap = sorted(test_indices.intersection(training_indices))
                if overlap:
                    raise RuntimeError(
                        f"{dataset} fixed test overlaps {key}: "
                        f"{overlap[:10]}"
                    )
                key_results[key] = {
                    "training_indices": len(training_indices),
                    "fixed_test_indices": len(test_indices),
                    "intersection": 0,
                }
            datasets[dataset] = key_results
    return {
        "status": "zero_source_index_intersection",
        "selection_reference": artifact(selection_reference_path),
        "datasets": datasets,
    }


def waveform_hash_reference(
    path: Path,
    *,
    expected_sha256: str,
    dataset: str,
    manifest_rows: list[dict[str, str]],
) -> dict[str, Any]:
    reference_path = reject_forbidden_path(path)
    if sha256_file(reference_path) != expected_sha256:
        raise RuntimeError(
            f"{dataset} waveform-hash reference SHA-256 differs"
        )
    with reference_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "dataset",
        "source_index",
        "trace_name",
        "event_id",
        "waveform_sha256",
    }
    if not rows or required.difference(rows[0]):
        raise RuntimeError(
            f"{dataset} waveform-hash reference lacks required columns"
        )
    expected_keys = {
        (row["source_index"], row["trace_name"], row["event_id"])
        for row in manifest_rows
    }
    hashes_by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        if row["dataset"] != dataset:
            raise RuntimeError(
                f"{dataset} waveform-hash reference contains another dataset"
            )
        key = (row["source_index"], row["trace_name"], row["event_id"])
        digest = row["waveform_sha256"]
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(
                f"{dataset} waveform-hash reference contains an invalid hash"
            )
        hashes_by_key.setdefault(key, set()).add(digest)
    if set(hashes_by_key) != expected_keys or any(
        len(values) != 1 for values in hashes_by_key.values()
    ):
        raise RuntimeError(
            f"{dataset} waveform-hash reference identities differ"
        )
    return {
        **artifact(reference_path),
        "records": len(expected_keys),
        "used_columns": [
            "dataset",
            "source_index",
            "trace_name",
            "event_id",
            "waveform_sha256",
        ],
        "ignored_columns": sorted(set(rows[0]).difference(required)),
        "evidence_role": (
            "pre-V27C selected-waveform content identity only; all prior "
            "model outputs and metrics are ignored"
        ),
    }


def hdf5_text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, str):
        return value
    raise RuntimeError(f"Expected HDF5 text scalar: {dataset.name}")


def source_input_contract(
    dataset: str,
    source_path: Path,
) -> dict[str, Any]:
    if dataset == "knet":
        return {
            "component_order": "ZNE",
            "dimension_order": "CW",
            "component_count": 3,
            "evidence": "fixed_record_manifest_component_order",
        }
    if dataset != "instancegm":
        raise ValueError(f"Unknown fixed dataset: {dataset}")
    with h5py.File(source_path, "r") as handle:
        if "data_format" not in handle:
            raise RuntimeError("InstanceGM source lacks data_format")
        data_format = handle["data_format"]
        for key in ("component_order", "dimension_order"):
            if key not in data_format:
                raise RuntimeError(
                    f"InstanceGM source lacks data_format/{key}"
                )
        component_order = hdf5_text(data_format["component_order"])
        dimension_order = hdf5_text(data_format["dimension_order"])
    if component_order != "ZNE" or dimension_order != "CW":
        raise RuntimeError(
            "InstanceGM source format differs: "
            f"component_order={component_order}, "
            f"dimension_order={dimension_order}"
        )
    return {
        "component_order": component_order,
        "dimension_order": dimension_order,
        "component_count": 3,
        "evidence": "source_store_data_format_group",
    }


def build_protocol(
    *,
    selection_freeze_path: Path,
    expected_selection_sha256: str,
    v24_protocol_path: Path,
    output_path: Path,
    run_root: Path,
    analysis_root: Path,
    allowed_output_root: Path = ALLOWED_OUTPUT_ROOT,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
    expected_development_protocol_sha256: str = (
        EXPECTED_DEVELOPMENT_PROTOCOL_SHA256
    ),
    expected_v24_protocol_sha256: str = EXPECTED_V24_PROTOCOL_SHA256,
    expected_training_selection_reference_sha256: str = (
        EXPECTED_TRAINING_SELECTION_REFERENCE_SHA256
    ),
    waveform_hash_reference_paths: Mapping[str, Path] = (
        DEFAULT_WAVEFORM_HASH_REFERENCES
    ),
    expected_waveform_hash_reference_sha256: Mapping[str, str] = (
        EXPECTED_WAVEFORM_HASH_REFERENCE_SHA256
    ),
    execution_amendment_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = require_output_path(
        output_path, allowed_root=allowed_output_root
    )
    run_output = require_output_path(
        run_root, allowed_root=allowed_output_root
    )
    analysis_output = require_output_path(
        analysis_root, allowed_root=allowed_output_root
    )
    if output.exists():
        raise FileExistsError(output)
    if run_output.exists() or analysis_output.exists():
        raise FileExistsError(
            "Retrospective score and analysis roots must not exist at freeze"
        )

    selection_path = reject_forbidden_path(selection_freeze_path)
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    expected_selection_sha256 = expected_selection_sha256.lower()
    if len(expected_selection_sha256) != 64:
        raise ValueError("--expected-selection-sha256 must have 64 hex digits")
    actual_selection_sha256 = sha256_file(selection_path)
    if actual_selection_sha256 != expected_selection_sha256:
        raise RuntimeError("GeoNet selection freeze SHA-256 differs")
    selection_freeze = load_json(selection_path)
    if selection_freeze.get("protocol_id") != SELECTION_FREEZE_ID:
        raise RuntimeError("Unexpected GeoNet selection freeze identifier")
    selected = resolve_selected_model(
        selection_freeze,
        expected_development_protocol_sha256=(
            expected_development_protocol_sha256
        ),
        expected_training_selection_reference_sha256=(
            expected_training_selection_reference_sha256
        ),
    )

    v24_path = reject_forbidden_path(v24_protocol_path)
    if sha256_file(v24_path) != expected_v24_protocol_sha256:
        raise RuntimeError("Canonical V24 fixed-test protocol SHA-256 differs")
    v24 = load_json(v24_path)
    if (
        v24.get("protocol_id")
        != "domain_robust_v24_station_level_external_test_v1"
    ):
        raise RuntimeError("Unexpected V24 fixed-test protocol")

    manifests: dict[str, dict[str, Any]] = {}
    manifest_rows: dict[str, list[dict[str, str]]] = {}
    cohorts: dict[str, dict[str, Any]] = {}
    for dataset in ("knet", "instancegm"):
        manifest_spec = v24["fixed_test_records"][dataset]
        manifest_path = verify_artifact(
            manifest_spec, require_sha256=True, require_mtime=False
        )
        rows = read_manifest(
            manifest_path,
            dataset=dataset,
            expected_count=int(expected_counts[dataset]),
        )
        manifest_rows[dataset] = rows
        manifests[dataset] = artifact(manifest_path)
        cohorts[dataset] = {
            "records": len(rows),
            "events": len({row["event_id"] for row in rows}),
            "stations": len({row["station_id"] for row in rows}),
            "selection": v24["fixed_test_cohorts"][dataset]["selection"],
        }
    overlap_audit = training_test_overlap_audit(
        Path(selected["training_selection_reference"]["path"]),
        manifest_rows,
    )
    waveform_hash_references = {
        dataset: waveform_hash_reference(
            waveform_hash_reference_paths[dataset],
            expected_sha256=(
                expected_waveform_hash_reference_sha256[dataset]
            ),
            dataset=dataset,
            manifest_rows=manifest_rows[dataset],
        )
        for dataset in ("knet", "instancegm")
    }

    source_stores: dict[str, dict[str, Any]] = {}
    waveform_input_contracts: dict[str, dict[str, Any]] = {}
    for dataset, key in (
        ("knet", "knet_waveforms"),
        ("instancegm", "instancegm_waveforms"),
    ):
        frozen_spec = v24["source_stores"][key]
        source_path = verify_artifact(
            frozen_spec,
            require_sha256=False,
            require_mtime=True,
        )
        source_stores[key] = {
            **artifact(source_path, hash_file=False),
            "identity_note": (
                "Bound by path, byte count, mtime, fixed record manifest, "
                "and a per-waveform SHA-256 emitted during scoring."
            ),
        }
        waveform_input_contracts[dataset] = source_input_contract(
            dataset, source_path
        )

    development_protocol = load_json(
        selected["development_protocol"]["path"]
    )
    runtime = dict(development_protocol["runtime"])
    required_runtime = {
        "sample_rate_hz": 100.0,
        "packet_samples": 50,
        "packet_duration_seconds": 0.5,
        "candidate_history_packets": 4,
        "max_arrival_age_packets": 4.0,
        "warmup_packets": 20,
    }
    for key, expected in required_runtime.items():
        actual = runtime.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, abs_tol=1e-12):
                raise RuntimeError(f"Frozen runtime differs: {key}")
        elif int(actual) != expected:
            raise RuntimeError(f"Frozen runtime differs: {key}")

    for relative in CODE_PATHS:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    baseline_methods = {
        "phasenet_instance_default": {
            "model_family": "PhaseNet",
            "endpoint": "same-manifest P-pick detection endpoint",
        },
        "eqtransformer_original_p_default": {
            "model_family": "EQTransformer",
            "endpoint": "same-manifest P-pick endpoint",
        },
        "eqtransformer_original_p_det_default": {
            "model_family": "EQTransformer",
            "endpoint": "same-manifest P-pick plus detection endpoint",
        },
    }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "evidence_status": (
            "geonet_selected_single_operating_point_applied_once_to_"
            "historically_inspected_fixed_retrospective_partitions"
        ),
        "selection_freeze": artifact(selection_path),
        "selection_result": selected["selection_result"],
        "development_protocol": selected["development_protocol"],
        "model": {
            "model_class": "CSPPCausalPickerV27C",
            "selected_arm": selected["selected_arm"],
            "primary_seed": selected["primary_seed"],
            "checkpoint": selected["checkpoint"],
            "training_selection_reference": selected[
                "training_selection_reference"
            ],
            "ensemble_members": 1,
        },
        "operating_point": {
            "candidate_threshold": selected["candidate_threshold"],
            "event_threshold": selected["event_threshold"],
            "count": 1,
            "threshold_scan_permitted": False,
            "provenance": (
                "GeoNet temporal development selection freeze; no fixed-"
                "benchmark threshold or model selection"
            ),
        },
        "runtime": runtime,
        "fixed_test_records": manifests,
        "fixed_test_cohorts": cohorts,
        "source_stores": source_stores,
        "selected_waveform_hash_references": waveform_hash_references,
        "waveform_input_contracts": waveform_input_contracts,
        "metrics": {
            "evaluation_unit": "one labeled three-component event record",
            "match_tolerance_seconds": 1.5,
            "hit_windows_seconds": [0.5, 1.0, 1.5],
            "local_p_snr_definition": "local_p_excess_power_1s_v1",
            "candidate_accounting": (
                "At most one candidate within 1.5 s is the record true "
                "positive; every remaining emitted candidate is an extra "
                "event-record candidate; an unmatched or technically failed "
                "record is a false negative. These event-record extra "
                "candidates are not continuous-stream false alarms."
            ),
            "candidate_precision_name": (
                "event_record_candidate_precision"
            ),
            "candidate_f1_name": "event_record_candidate_f1",
            "technical_failure_policy": "count_as_miss",
            "test_set_model_or_threshold_selection": False,
        },
        "statistics": {
            "bootstrap_seed": 20260729,
            "bootstrap_replicates": 5000,
            "bootstrap_cluster": "event_id",
            "snr_bins_db": ["<0", "0-<10", ">=10", "invalid"],
            "knet_p_precontext_bins_seconds": ["<10", ">=10"],
        },
        "public_baseline_endpoint_interface": {
            "status": "reserved_no_artifact_bound_at_protocol_freeze",
            "methods": baseline_methods,
            "required_record_identity": [
                "dataset",
                "source_index",
                "trace_name",
                "event_id",
            ],
            "required_endpoint": "true_positive_at_1p5_seconds",
            "same_manifest_required": True,
            "latency_comparison_permitted": False,
            "missing_artifact_policy": (
                "omit comparison; never synthesize or backfill baseline rows"
            ),
        },
        "execution": {
            "device": "cpu",
            "torch_threads": 1,
            "workers": 1,
            "fresh_process_per_dataset": True,
            "score_output_root": str(run_output),
            "analysis_output_root": str(analysis_output),
            "allowed_output_root": str(allowed_output_root.resolve()),
            "forbidden_volume": str(FORBIDDEN_VOLUME),
            "global_lock_path": str(
                allowed_output_root.resolve()
                / "locks/cspp_v27_heavy_evaluation.lock"
            ),
            "exclusive_process_patterns": [
                "run_v27_geonet_temporal_score_workers.py",
                "build_v27_geonet_temporal_score_cache.py",
                "run_v27_geonet_final_score_workers.py",
                "build_v27_geonet_final_score_cache.py",
                "run_cspp_v27c_fixed_retrospective.py",
                "eval_cspp_v27c_fixed_retrospective.py",
                "run_v27c_fixed_public_baselines.py",
                "eval_v27c_fixed_public_baselines.py",
            ],
            "maximum_worker_resident_set_gib": 8.0,
            "minimum_system_available_memory_gib": 8.0,
            "maximum_system_swap_used_gib": 4.0,
            "memory_guard_poll_seconds": 0.25,
        },
        "v24_fixed_test_protocol": artifact(v24_path),
        "training_test_overlap_audit": overlap_audit,
        "code_sha256": {
            relative: sha256_file(ROOT / relative)
            for relative in CODE_PATHS
        },
        "claim_boundary": {
            "historical_status": (
                "K-NET and InstanceGM are historically inspected fixed "
                "retrospective benchmark partitions, not new blind tests."
            ),
            "supported_generalization": (
                "The benchmark reports held-out record-level performance "
                "on the frozen partitions. Event-level independence from "
                "training has not been established."
            ),
            "station_boundary": (
                "Training and fixed cohorts share stations; results do not "
                "establish unseen-station generalization."
            ),
            "false_alarm_boundary": (
                "Event-centered records do not estimate continuous-stream "
                "false-alarm rates or false alarms per station-day."
            ),
            "public_baseline_boundary": (
                "Optional public-model endpoint CSVs remain user-supplied "
                "same-manifest comparisons unless a separate generation "
                "provenance seal binds weights, code, preprocessing, and "
                "output. Unsealed CSVs cannot support a formal SOTA claim. "
                "Released offline interfaces are not used for causal "
                "availability-latency comparison."
            ),
            "input_identity_boundary": (
                "The large source stores are bound by path, byte count, and "
                "mtime. Every successfully decoded selected waveform must "
                "also match a pre-V27C per-record SHA-256 reference; old "
                "model outputs in those reference CSVs are ignored."
            ),
        },
    }
    if execution_amendment_record is not None:
        payload["execution_amendment"] = dict(
            execution_amendment_record
        )
    return payload


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    amendment = None
    if args.parent_protocol is not None:
        amendment = execution_amendment(
            parent_protocol_path=args.parent_protocol,
            expected_parent_protocol_sha256=(
                args.expected_parent_protocol_sha256
            ),
            failed_attempt_root=args.failed_attempt_root,
        )
    payload = build_protocol(
        selection_freeze_path=args.selection_freeze,
        expected_selection_sha256=args.expected_selection_sha256,
        v24_protocol_path=args.v24_protocol,
        output_path=args.output,
        run_root=args.run_root,
        analysis_root=args.analysis_root,
        execution_amendment_record=amendment,
    )
    output = require_output_path(args.output)
    write_json_once(output, payload)
    print(
        json.dumps(
            {
                "protocol": artifact(output),
                "selected_arm": payload["model"]["selected_arm"],
                "event_threshold": payload["operating_point"][
                    "event_threshold"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
