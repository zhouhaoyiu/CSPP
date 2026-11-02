#!/usr/bin/env python3
"""Freeze same-record public-model baselines for the GeoNet 2026 holdout."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import v27_runner_lease  # noqa: E402
from src.v27_released_public_full_record import (  # noqa: E402
    EDGE_GAP_FILL_VALUE,
    EDGE_GAP_MAX_FRACTION,
    EDGE_GAP_MAX_SAMPLES,
    released_annotation_preprocessing_contract,
)


WORK_ROOT = Path("workspace").resolve()
FORBIDDEN_ROOT = Path("external_volume").resolve()
FINAL_PROTOCOL_ID = "cspp_v27c_geonet_final_evaluation_v1"
FINAL_ANALYSIS_PROTOCOL_ID = "cspp_v27c_geonet_final_analysis_a2"
FINAL_ANALYSIS_STATUS = (
    "transparent_post_event_cache_statistical_amendment_before_complete_"
    "final_analysis_without_rescoring_or_operating_point_change"
)
SOURCE_PUBLIC_PROTOCOL_ID = "domain_robust_v24_station_level_external_test_v1"
PROTOCOL_ID = "cspp_v27c_geonet_final_public_baselines_v2"
SUPERSEDED_PROTOCOL_ID = "cspp_v27c_geonet_final_public_baselines_v1"
DEFAULT_FINAL_PROTOCOL = (
    WORK_ROOT / "protocol_freeze/"
    "cspp_v27c_geonet_final_evaluation_protocol_20260729.json"
)
DEFAULT_SOURCE_PUBLIC_PROTOCOL = (
    ROOT / "experiments/v24_station_level_external_test_20260728/"
    "station_level_test_protocol.json"
)
DEFAULT_FINAL_ANALYSIS_RESULT = (
    WORK_ROOT
    / "evaluation/"
    "cspp_v27c_geonet_final_analysis_a2_20260729/result.json"
)
DEFAULT_OUTPUT = (
    WORK_ROOT / "protocol_freeze/"
    "cspp_v27c_geonet_final_public_baseline_protocol_v2_20260729.json"
)
DEFAULT_RUN_ROOT = (
    WORK_ROOT
    / "evaluation/cspp_v27c_geonet_final_public_baselines_v2_20260729/runs"
)
DEFAULT_ANALYSIS_ROOT = (
    WORK_ROOT
    / "evaluation/cspp_v27c_geonet_final_public_baselines_v2_20260729/analysis"
)
DEFAULT_SUPERSEDED_PROTOCOL = (
    WORK_ROOT
    / "protocol_freeze/"
    "cspp_v27c_geonet_final_public_baseline_protocol_20260729.json"
)
PRIMARY_LABEL_SOURCE = "geonet_origin_associated_used_p_pick"
PRIMARY_LABEL_STRATUM = "authoritative_used_p_positive_weight"
EXPECTED_MODELS = {"phasenet_instance", "eqtransformer_original"}
SOURCE_EXPECTED_METHODS = {
    "phasenet_instance": {"phasenet_instance_default"},
    "eqtransformer_original": {
        "eqtransformer_original_p_default",
        "eqtransformer_original_p_det_default",
    },
}
SOURCE_TO_OUTPUT_METHOD = {
    "eqtransformer_original_p_det_default": (
        "eqtransformer_original_p_det_custom"
    ),
}
EXPECTED_METHODS = {
    "phasenet_instance": {"phasenet_instance_default"},
    "eqtransformer_original": {
        "eqtransformer_original_p_default",
        "eqtransformer_original_p_det_custom",
    },
}
EXPECTED_OPERATING_POINTS = {
    "phasenet_instance_default": {
        "p_threshold": 0.08639824281948567,
        "detection_threshold": None,
    },
    "eqtransformer_original_p_default": {
        "p_threshold": 0.1,
        "detection_threshold": None,
    },
    "eqtransformer_original_p_det_custom": {
        "p_threshold": 0.1,
        "detection_threshold": 0.3,
    },
}
CODE_PATHS = (
    "scripts/evaluation/freeze_v27_geonet_final_public_baseline_protocol.py",
    "scripts/evaluation/eval_v27_geonet_final_public_baselines.py",
    "scripts/evaluation/run_v27_geonet_final_public_baselines.py",
    "scripts/evaluation/analyze_v27_geonet_final_public_baselines.py",
    "scripts/evaluation/analyze_v27_geonet_temporal_selection.py",
    "scripts/evaluation/v27_runner_lease.py",
    "scripts/evaluation/freeze_v27c_fixed_public_baseline_protocol.py",
    "scripts/evaluation/eval_v27c_fixed_public_baselines.py",
    "scripts/evaluation/eval_cspp_v27c_fixed_retrospective.py",
    "scripts/evaluation/eval_v24_fixed_test_models.py",
    "scripts/evaluation/run_v20c_geonet_score_cache_safe.py",
    "scripts/evaluation/build_v20c_geonet_score_cache.py",
    "scripts/evaluation/build_v27_geonet_temporal_score_cache.py",
    "src/causal_fir_decimator.py",
    "src/causal_packet_veto_features.py",
    "src/causal_stream_veto_features.py",
    "src/config.py",
    "src/cspp_v27_reused_core_score_cache.py",
    "src/current_packet_artifact_features.py",
    "src/current_packet_artifact_head.py",
    "src/data_streaming.py",
    "src/domain_robust_model_io.py",
    "src/v20c_public_window_baselines.py",
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
    "src/v27_released_public_full_record.py",
    "src/v15_reused_packet_features.py",
    "src/v20c_causal_score_cache.py",
    "src/v20c_event_snr.py",
    "src/v21_causal_score_cache.py",
    "src/v21_score_shape_calibrator.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reject_forbidden_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved == FORBIDDEN_ROOT or resolved.is_relative_to(FORBIDDEN_ROOT):
        raise RuntimeError(f"WD_BLACK is prohibited: {resolved}")
    return resolved


def require_output_path(
    path: str | Path,
    *,
    allowed_root: Path = WORK_ROOT,
) -> Path:
    resolved = reject_forbidden_path(path)
    allowed = reject_forbidden_path(allowed_root)
    if resolved != allowed and not resolved.is_relative_to(allowed):
        raise RuntimeError(f"Output escapes the allowed root: {resolved}")
    return resolved


def verify_artifact(spec: Mapping[str, Any]) -> Path:
    path = reject_forbidden_path(str(spec["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Frozen artifact differs: {path}")
    return path


def load_bound_json(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_protocol_id: str,
) -> tuple[dict[str, Any], str, Path]:
    resolved = reject_forbidden_path(path)
    actual = sha256_file(resolved)
    if actual != expected_sha256.lower():
        raise RuntimeError(f"Protocol SHA-256 differs: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != expected_protocol_id:
        raise RuntimeError(f"Unexpected protocol identifier: {resolved}")
    return payload, actual, resolved


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def station_key(row: Mapping[str, Any]) -> str:
    return ".".join((str(row["network"]), str(row["station"]), str(row["location"])))


def _finite_float(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"{label} must be finite")
    return parsed


def _verify_final_protocol(
    final: Mapping[str, Any],
    *,
    final_path: Path,
    final_sha256: str,
) -> None:
    if final.get("status") != (
        "one_operating_point_and_three_predeclared_seeds_frozen_"
        "before_final_score_access"
    ):
        raise RuntimeError("Final protocol is not post-selection frozen")
    point = final.get("operating_point", {})
    if (
        int(point.get("count", -1)) != 1
        or point.get("threshold_scan_permitted") is not False
        or point.get("model_or_threshold_selection_on_final_holdout") is not False
    ):
        raise RuntimeError("Final protocol does not seal one operating point")
    selection = verify_artifact(final["selection_freeze"])
    selection_payload = json.loads(selection.read_text(encoding="utf-8"))
    if (
        selection_payload.get("protocol_id")
        != "cspp_v27_geonet_temporal_selection_freeze_v1"
        or selection_payload.get("status")
        != (
            "development_arm_threshold_and_primary_seed_"
            "irrevocably_frozen"
        )
    ):
        raise RuntimeError("Selection-freeze artifact is not final")
    selection_point = selection_payload.get("operating_point", {})
    for field in (
        "selected_arm",
        "candidate_threshold",
        "event_threshold",
    ):
        if str(selection_point.get(field)) != str(point.get(field)):
            raise RuntimeError(f"Selection and final operating points differ: {field}")
    required_inputs = {
        "event_manifest_protocol",
        "event_scorable_stations",
        "event_response_files",
        "orientation_manifest_protocol",
        "event_station_orientations",
    }
    if required_inputs.difference(final.get("inputs", {})):
        raise RuntimeError("Final protocol lacks frozen event inputs")
    for name in required_inputs:
        verify_artifact(final["inputs"][name])
    event_manifest_path = verify_artifact(final["inputs"]["event_manifest_protocol"])
    event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
    if event_manifest.get("protocol_id") != (
        "cspp_v27_geonet_final_event_waveform_manifest_v1"
    ):
        raise RuntimeError("Final event waveform manifest is not frozen")
    orientation_manifest_path = verify_artifact(
        final["inputs"]["orientation_manifest_protocol"]
    )
    orientation_manifest = json.loads(
        orientation_manifest_path.read_text(encoding="utf-8")
    )
    if orientation_manifest.get("protocol_id") != (
        "cspp_v27_geonet_final_orientation_manifest_v1"
    ):
        raise RuntimeError("Final orientation manifest is not frozen")
    artifact_pairs = (
        (
            event_manifest.get("artifacts", {}).get("scorable_stations"),
            final["inputs"]["event_scorable_stations"],
        ),
        (
            event_manifest.get("artifacts", {}).get("scorable_response_files"),
            final["inputs"]["event_response_files"],
        ),
        (
            orientation_manifest.get("artifacts", {}).get("event_station_orientations"),
            final["inputs"]["event_station_orientations"],
        ),
    )
    for nested, bound in artifact_pairs:
        if not isinstance(nested, Mapping) or any(
            str(nested.get(field)) != str(bound.get(field))
            for field in ("path", "bytes", "sha256")
        ):
            raise RuntimeError("Final nested manifest artifact differs")
    code_hashes = final.get("code_sha256")
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise RuntimeError("Final protocol lacks code identities")
    for relative, expected in code_hashes.items():
        source = ROOT / str(relative)
        if not source.is_file() or sha256_file(source) != str(expected):
            raise RuntimeError(f"Final protocol code drifted: {relative}")
    if sha256_file(final_path) != final_sha256:
        raise RuntimeError("Final protocol changed during verification")


def _orientation_groups(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    for group_id, values in groups.items():
        values.sort(key=lambda item: int(item["component_index"]))
        indexes = [int(item["component_index"]) for item in values]
        if len(values) != 3 or indexes not in ([0, 1, 2], [1, 2, 3]):
            raise RuntimeError(
                "Orientation group does not have one complete consecutive "
                f"three-component index set: {group_id}"
            )
    return groups


def build_record_manifest(
    final: Mapping[str, Any],
) -> list[dict[str, Any]]:
    station_path = verify_artifact(final["inputs"]["event_scorable_stations"])
    response_path = verify_artifact(final["inputs"]["event_response_files"])
    orientation_path = verify_artifact(final["inputs"]["event_station_orientations"])
    stations = read_csv(station_path)
    responses = read_csv(response_path)
    orientation_groups = _orientation_groups(read_csv(orientation_path))
    response_by_event: dict[str, dict[str, str]] = {}
    for row in responses:
        event_id = row["event_id"]
        if event_id in response_by_event:
            raise RuntimeError(f"Duplicate event response: {event_id}")
        response_file = reject_forbidden_path(row["response_file"])
        if (
            not response_file.is_file()
            or response_file.stat().st_size != int(row["response_bytes"])
            or sha256_file(response_file) != row["response_sha256"]
        ):
            raise RuntimeError(f"Event response differs: {event_id}")
        response_by_event[event_id] = row

    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in stations:
        if row["label_source"] != PRIMARY_LABEL_SOURCE:
            continue
        weight = _finite_float(row["pick_time_weight"], "pick_time_weight")
        if weight <= 0.0:
            raise RuntimeError("Authoritative used-P record has nonpositive weight")
        event_id = row["event_id"]
        key = station_key(row)
        identity = (event_id, key)
        if identity in identities:
            raise RuntimeError(f"Duplicate final public-baseline record: {identity}")
        identities.add(identity)
        response = response_by_event.get(event_id)
        if response is None:
            raise RuntimeError(f"Scorable station lacks event response: {identity}")
        group_id = f"{event_id}.{key}"
        orientation_rows = orientation_groups.get(group_id)
        if orientation_rows is None:
            raise RuntimeError(f"Scorable station lacks orientation: {identity}")
        channels = [item["channel"] for item in orientation_rows]
        if channels != row["component_set"].split(";"):
            raise RuntimeError(f"Component order differs: {identity}")
        reference = str(row["p_arrival_time_utc"])
        if not reference:
            raise RuntimeError(f"Reference P is absent: {identity}")
        record = {
            "record_id": f"{event_id}/{key}",
            "event_id": event_id,
            "station_key": key,
            "network": row["network"],
            "station": row["station"],
            "location": row["location"],
            "site_rank": int(row["site_rank"]),
            "label_source": row["label_source"],
            "label_stratum": PRIMARY_LABEL_STRATUM,
            "pick_time_weight": weight,
            "reference_p_time_utc": reference,
            "origin_time_utc": response["origin_time_utc"],
            "magnitude": _finite_float(row["magnitude"], "magnitude"),
            "epicentral_distance_km": _finite_float(
                row["epicentral_distance_km"],
                "epicentral_distance_km",
            ),
            "component_set": row["component_set"],
            "response_file": str(reject_forbidden_path(response["response_file"])),
            "response_bytes": int(response["response_bytes"]),
            "response_sha256": response["response_sha256"],
            "orientation_group_id": group_id,
            "orientation_group_sha256": stable_digest(orientation_rows),
        }
        records.append(record)
    records.sort(
        key=lambda item: (
            item["origin_time_utc"],
            item["event_id"],
            item["site_rank"],
            item["station_key"],
        )
    )
    if not records:
        raise RuntimeError("No positive-weight authoritative final records")
    return records


def tensor_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _loaded_state_dict_sha256(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        raise RuntimeError(f"Unexpected public-model weight payload: {path}")
    return tensor_state_digest(payload)


def collect_public_models(
    source: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    models = source.get("public_models", {})
    if set(models) != EXPECTED_MODELS:
        raise RuntimeError("Source public-model set differs")
    output: dict[str, dict[str, Any]] = {}
    for model_key in sorted(EXPECTED_MODELS):
        spec = dict(models[model_key])
        points = spec.get("operating_points", [])
        observed_methods = {str(point["id"]) for point in points}
        if observed_methods != SOURCE_EXPECTED_METHODS[model_key]:
            raise RuntimeError(f"Public operating points differ: {model_key}")
        methods = []
        for point in points:
            source_method_id = str(point["id"])
            method_id = SOURCE_TO_OUTPUT_METHOD.get(
                source_method_id,
                source_method_id,
            )
            p_threshold = _finite_float(
                point["p_threshold"], f"{model_key}.p_threshold"
            )
            detection = point.get("detection_threshold")
            if detection is not None:
                detection = _finite_float(detection, f"{model_key}.detection_threshold")
            expected_point = EXPECTED_OPERATING_POINTS[method_id]
            if (
                p_threshold != expected_point["p_threshold"]
                or detection != expected_point["detection_threshold"]
            ):
                raise RuntimeError(
                    f"Public operating point values differ: {method_id}"
                )
            methods.append(
                {
                    **dict(point),
                    "id": method_id,
                    "source_method_id": source_method_id,
                    "p_threshold": p_threshold,
                    "detection_threshold": detection,
                    "threshold_count": 1,
                    "holdout_tuned": False,
                    "endpoint_role": (
                        "custom_p_peak_detection_score_gate_diagnostic"
                        if method_id
                        == "eqtransformer_original_p_det_custom"
                        else "released_p_pick_endpoint"
                    ),
                    "official_default_endpoint": (
                        method_id
                        != "eqtransformer_original_p_det_custom"
                    ),
                }
            )
        artifacts = {
            name: artifact(verify_artifact(value))
            for name, value in spec["artifacts"].items()
        }
        filter_spec = spec.get("trailing_window_filter")
        preprocessing = released_annotation_preprocessing_contract(spec)
        output[model_key] = {
            **spec,
            "operating_points": methods,
            "artifacts": artifacts,
            "trailing_window_filter": filter_spec,
            "released_annotation_preprocessing": preprocessing,
            "loaded_state_dict_sha256": _loaded_state_dict_sha256(
                Path(artifacts["weights"]["path"])
            ),
        }
    return output


def verify_final_analysis_result(
    path: Path,
    *,
    expected_sha256: str,
    final_protocol_path: Path,
    final_protocol_sha256: str,
    primary_seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    resolved = reject_forbidden_path(path)
    result_spec = artifact(resolved)
    if result_spec["sha256"] != expected_sha256.lower():
        raise RuntimeError("Final A2 analysis result SHA-256 differs")
    result = json.loads(resolved.read_text(encoding="utf-8"))
    protocol_spec = result.get("protocol", {})
    if (
        result.get("result_id")
        != "cspp_v27c_geonet_final_evaluation_result_a2"
        or Path(str(protocol_spec.get("path", ""))).resolve()
        != final_protocol_path.resolve()
        or str(protocol_spec.get("sha256", ""))
        != final_protocol_sha256
        or verify_artifact(protocol_spec) != final_protocol_path.resolve()
        or int(
            result.get("seed_policy", {}).get(
                "primary_operational_seed", -1
            )
        )
        != primary_seed
    ):
        raise RuntimeError("Final A2 analysis result identity differs")
    analysis_spec = result.get("analysis_protocol", {})
    analysis_path = verify_artifact(analysis_spec)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis_final_spec = analysis.get("final_evaluation_protocol", {})
    disclosure = analysis.get("analysis_amendment_disclosure")
    if (
        result.get("analysis_protocol_sha256")
        != analysis_spec.get("sha256")
        or analysis.get("protocol_id") != FINAL_ANALYSIS_PROTOCOL_ID
        or analysis.get("status") != FINAL_ANALYSIS_STATUS
        or Path(str(analysis_final_spec.get("path", ""))).resolve()
        != final_protocol_path.resolve()
        or str(analysis_final_spec.get("sha256", ""))
        != final_protocol_sha256
        or verify_artifact(analysis_final_spec) != final_protocol_path.resolve()
        or not isinstance(disclosure, Mapping)
        or result.get("analysis_amendment_disclosure") != disclosure
    ):
        raise RuntimeError("Final A2 statistical amendment binding differs")
    station_spec = result.get("artifacts", {}).get(
        "event_station_results", {}
    )
    verify_artifact(station_spec)
    return result, result_spec, dict(station_spec), dict(analysis_spec)


def build_protocol(
    *,
    final_protocol_path: Path,
    expected_final_protocol_sha256: str,
    final_analysis_result_path: Path,
    expected_final_analysis_result_sha256: str,
    source_public_protocol_path: Path,
    expected_source_public_protocol_sha256: str,
    output_path: Path,
    run_root: Path,
    analysis_root: Path,
    superseded_protocol_path: Path | None = None,
    expected_superseded_protocol_sha256: str | None = None,
    allowed_output_root: Path = WORK_ROOT,
) -> dict[str, Any]:
    output = require_output_path(output_path, allowed_root=allowed_output_root)
    runs = require_output_path(run_root, allowed_root=allowed_output_root)
    analysis = require_output_path(analysis_root, allowed_root=allowed_output_root)
    for path in (output, runs, analysis):
        if path.exists():
            raise FileExistsError(path)
    final, final_sha256, final_path = load_bound_json(
        final_protocol_path,
        expected_sha256=expected_final_protocol_sha256,
        expected_protocol_id=FINAL_PROTOCOL_ID,
    )
    _verify_final_protocol(
        final,
        final_path=final_path,
        final_sha256=final_sha256,
    )
    source, source_sha256, source_path = load_bound_json(
        source_public_protocol_path,
        expected_sha256=expected_source_public_protocol_sha256,
        expected_protocol_id=SOURCE_PUBLIC_PROTOCOL_ID,
    )
    records = build_record_manifest(final)
    (
        final_result,
        final_result_spec,
        final_station_spec,
        final_analysis_protocol_spec,
    ) = verify_final_analysis_result(
        final_analysis_result_path,
        expected_sha256=expected_final_analysis_result_sha256,
        final_protocol_path=final_path,
        final_protocol_sha256=final_sha256,
        primary_seed=int(
            final["seed_policy"]["primary_operational_seed"]
        ),
    )
    public_models = collect_public_models(source)
    for relative in CODE_PATHS:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    methods = [
        {
            "model_key": model_key,
            **point,
        }
        for model_key, spec in sorted(public_models.items())
        for point in spec["operating_points"]
    ]
    superseded_protocol = None
    if superseded_protocol_path is not None:
        if expected_superseded_protocol_sha256 is None:
            raise RuntimeError("Superseded protocol SHA-256 is required")
        _prior, _prior_sha, prior_path = load_bound_json(
            superseded_protocol_path,
            expected_sha256=expected_superseded_protocol_sha256,
            expected_protocol_id=SUPERSEDED_PROTOCOL_ID,
        )
        superseded_protocol = artifact(prior_path)
    return {
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "post_hoc_same_record_geonet_2026_public_benchmark_with_"
            "pretuned_weights_thresholds_and_disclosed_edge_gap_remediation"
        ),
        "input_remediation_amendment": {
            "prespecification_status": (
                "post_hoc_after_v1_input_validation_failure_before_any_"
                "public_model_result_was_published"
            ),
            "superseded_v1_protocol": superseded_protocol,
            "observed_v1_failure": (
                "The first PhaseNet execution stopped before publication "
                "because the oriented constructor retained sparse NaNs at "
                "the left resampling edge."
            ),
            "unchanged": (
                "public weights, operating thresholds, record cohort, "
                "reference arrivals, matching endpoint, and statistics"
            ),
        },
        "final_evaluation_protocol": artifact(final_path),
        "final_evaluation_protocol_sha256": final_sha256,
        "selection_freeze": final["selection_freeze"],
        "frozen_final_inputs": {
            name: final["inputs"][name]
            for name in (
                "event_manifest_protocol",
                "event_scorable_stations",
                "event_response_files",
                "orientation_manifest_protocol",
                "event_station_orientations",
            )
        },
        "source_public_model_protocol": artifact(source_path),
        "source_public_model_protocol_sha256": source_sha256,
        "public_models": public_models,
        "methods": methods,
        "record_manifest": records,
        "record_manifest_sha256": stable_digest(records),
        "record_count": len(records),
        "event_count": len({record["event_id"] for record in records}),
        "endpoint": {
            "evaluation_unit": (
                "technically scorable GeoNet event-station record with an "
                "authoritative positive-weight P reference"
            ),
            "label_source": PRIMARY_LABEL_SOURCE,
            "label_stratum": PRIMARY_LABEL_STRATUM,
            "reference": "the same frozen GeoNet P arrival used by CSPP",
            "match_tolerance_seconds": 1.5,
            "hit_windows_seconds": [0.5, 1.0, 1.5],
            "candidate_accounting": (
                "At most one nearest candidate inside +/-1.5 s is the record "
                "true positive. Every other candidate is an event-record "
                "false positive; when no candidate is inside the tolerance, "
                "all candidates are false positives."
            ),
            "tie_breaking": (
                "Minimum absolute arrival error, then predicted arrival time "
                "and candidate index for the offline public models. CSPP uses "
                "availability time after minimum absolute arrival error."
            ),
            "interface": "released SeisBench full-record offline annotation",
            "preprocessing": (
                "Each complete 180 s oriented three-component record enters "
                "annotate once. Any released weight-specific filter is owned "
                "and applied inside SeisBench; no external prefilter is "
                "permitted. Before annotation, sparse nonfinite samples are "
                "filled only under the separately frozen edge-gap contract. "
                "EQTransformer's zero-phase filter uses future samples within "
                "the complete record."
            ),
            "edge_gap_remediation": {
                "maximum_edge_samples_per_side": EDGE_GAP_MAX_SAMPLES,
                "maximum_nonfinite_component_sample_fraction": (
                    EDGE_GAP_MAX_FRACTION
                ),
                "fill_value": EDGE_GAP_FILL_VALUE,
                "scope": "first_or_last_one_second_only",
                "role": (
                    "post_hoc_input_adapter_correction_after_v1_failed_"
                    "before_result_publication"
                ),
                "internal_or_larger_gap_policy": "retain_as_technical_miss",
            },
            "record_interval": "origin_time_minus_60_s_to_plus_120_s",
            "technical_failure_policy": (
                "A source-record decode failure or oriented-waveform "
                "construction ValueError/RuntimeError is retained as a miss "
                "for descriptive accounting. Frozen-artifact, orientation, "
                "shape, model-loading, preprocessing-contract, inference, or "
                "method-output errors abort the complete evaluation. Any "
                "retained technical failure makes the affected method and "
                "paired comparison non-inferential."
            ),
            "availability_latency_generated": False,
            "availability_latency_comparison_permitted": False,
            "test_set_threshold_tuning_permitted": False,
        },
        "statistics": {
            "confidence_level": 0.95,
            "bootstrap_cluster": "event_id",
            "station_cluster_sensitivity": "station_key",
            "bootstrap_seed": 20260729,
            "bootstrap_replicates": 5000,
            "paired_difference_direction": "public_model_minus_cspp",
            "mcnemar_role": "not_implemented",
            "technical_failure_interval_policy": (
                "retain descriptive point estimates and set interval bounds "
                "to null for the affected method and paired comparison"
            ),
        },
        "comparison_timing": {
            "weights_and_thresholds": (
                "released weights and method thresholds were fixed before "
                "GeoNet 2026 public-model scoring and were not tuned on the "
                "holdout"
            ),
            "endpoint_and_statistics": (
                "same-record endpoint and uncertainty plan were frozen post "
                "hoc after the CSPP A2 result existed and before any released "
                "public model was run on these records"
            ),
            "prespecification_status": (
                "post_hoc_same_record_public_benchmark_with_pretuned_"
                "public_model_operating_points"
            ),
        },
        "cspp_comparator": {
            "protocol_sha256": final_sha256,
            "primary_operational_seed": int(
                final["seed_policy"]["primary_operational_seed"]
            ),
            "result": final_result_spec,
            "analysis_protocol": final_analysis_protocol_spec,
            "analysis_amendment_disclosure": final_result[
                "analysis_amendment_disclosure"
            ],
            "event_station_results": final_station_spec,
            "same_record_keys_required": True,
        },
        "execution": {
            "python_executable": str(Path(sys.executable).resolve()),
            "device": "mps",
            "workers": 1,
            "torch_threads": 1,
            "fresh_process_per_model": True,
            "serial_models": True,
            "run_root": str(runs),
            "analysis_root": str(analysis),
            "allowed_output_root": str(reject_forbidden_path(allowed_output_root)),
            "forbidden_volume": str(FORBIDDEN_ROOT),
            "global_lock_path": str(
                reject_forbidden_path(allowed_output_root)
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
                "run_v27_geonet_final_public_baselines.py",
                "eval_v27_geonet_final_public_baselines.py",
            ],
            "maximum_worker_resident_set_gib": 8.0,
            "minimum_system_available_memory_gib": 8.0,
            "maximum_system_swap_used_gib": 4.0,
            "memory_guard_poll_seconds": 0.25,
            "runner_lease_id": v27_runner_lease.LEASE_ID,
            "atomic_publication": (
                "runner publishes lease, records, summary, and audit by one "
                "attempt-directory rename"
            ),
        },
        "software_versions": {
            package: version(package)
            for package in ("numpy", "obspy", "seisbench", "torch")
        },
        "code_sha256": {
            relative: sha256_file(ROOT / relative) for relative in CODE_PATHS
        },
        "claim_boundary": {
            "comparison_scope": (
                "PhaseNet-INSTANCE and EQTransformer-original are evaluated "
                "on the same GeoNet 2026 technically scorable station "
                "records, positive-weight GeoNet P references associated "
                "with the preferred origin, and "
                "+/-1.5 s record endpoint used for CSPP."
            ),
            "offline_only": (
                "The public models use full-record offline annotation. This "
                "supports record-level endpoint comparison only."
            ),
            "causal_or_online_fairness": (
                "No online, causal-availability, packet-latency, or "
                "instrument-runtime fairness claim is permitted."
            ),
            "holdout_status": (
                "GeoNet 2026 waveform access follows CSPP arm, threshold, and "
                "seed freeze; public weights and method thresholds are fixed "
                "without GeoNet 2026 tuning."
            ),
            "comparison_timing": (
                "The same-record released-model benchmark is post hoc: its "
                "endpoint and statistical plan were frozen after the CSPP A2 "
                "result existed. The sparse edge-gap input rule was added "
                "after the first v1 run failed before publishing predictions. "
                "Only the released weights and operating thresholds were "
                "fixed independently of this comparison."
            ),
            "eqtransformer_detection_gate": (
                "EQTransformer P-only is the released P-pick endpoint. The "
                "P+Detection variant is a prespecified custom diagnostic that "
                "requires the detection score at the P-segment peak to reach "
                "0.3; it is not SeisBench's default detection endpoint."
            ),
            "cspp_statistical_amendment": final_result[
                "analysis_amendment_disclosure"
            ]["claim_boundary"],
            "technical_failures": (
                "Retained input decode or waveform-construction failures "
                "invalidate formal inference for the affected public method "
                "and its paired CSPP comparison. Model-contract, "
                "preprocessing, or inference failures abort evaluation."
            ),
            "candidate_precision": (
                "Public-model event-record candidate precision is a "
                "descriptive quantity conditional on emitted candidates from "
                "successfully processed records. Technical success is "
                "reported separately."
            ),
        },
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-protocol", type=Path, default=DEFAULT_FINAL_PROTOCOL)
    parser.add_argument("--expected-final-protocol-sha256", required=True)
    parser.add_argument(
        "--final-analysis-result",
        type=Path,
        default=DEFAULT_FINAL_ANALYSIS_RESULT,
    )
    parser.add_argument(
        "--expected-final-analysis-result-sha256",
        required=True,
    )
    parser.add_argument(
        "--source-public-protocol",
        type=Path,
        default=DEFAULT_SOURCE_PUBLIC_PROTOCOL,
    )
    parser.add_argument(
        "--expected-source-public-protocol-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--superseded-protocol",
        type=Path,
        default=DEFAULT_SUPERSEDED_PROTOCOL,
    )
    parser.add_argument(
        "--expected-superseded-protocol-sha256",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_protocol(
        final_protocol_path=args.final_protocol,
        expected_final_protocol_sha256=args.expected_final_protocol_sha256,
        final_analysis_result_path=args.final_analysis_result,
        expected_final_analysis_result_sha256=(
            args.expected_final_analysis_result_sha256
        ),
        source_public_protocol_path=args.source_public_protocol,
        expected_source_public_protocol_sha256=(
            args.expected_source_public_protocol_sha256
        ),
        output_path=args.output,
        run_root=args.run_root,
        analysis_root=args.analysis_root,
        superseded_protocol_path=args.superseded_protocol,
        expected_superseded_protocol_sha256=(
            args.expected_superseded_protocol_sha256
        ),
    )
    output = require_output_path(args.output)
    write_json_once(output, payload)
    print(json.dumps(artifact(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
