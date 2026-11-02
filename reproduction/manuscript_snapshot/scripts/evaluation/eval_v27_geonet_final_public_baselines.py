#!/usr/bin/env python3
"""Evaluate one sealed public picker on the GeoNet 2026 final records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping

import numpy as np
from obspy import UTCDateTime
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import (  # noqa: E402
    eval_v24_fixed_test_models as legacy,
)
from scripts.evaluation import (  # noqa: E402
    eval_v27c_fixed_public_baselines as sealed_public,
)
from scripts.evaluation import (  # noqa: E402
    freeze_v27_geonet_final_public_baseline_protocol as freeze,
)
from scripts.evaluation import (  # noqa: E402
    v27_runner_lease as runner_lease,
)
from scripts.evaluation.build_v27_geonet_temporal_score_cache import (  # noqa: E402
    channel_orientations,
    station_target,
)
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    stream_to_oriented_gappy_zne_array,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402
from v27_released_public_full_record import (  # noqa: E402
    fill_small_edge_gaps,
    score_released_full_record,
    validate_released_annotation_preprocessing,
)


DEFAULT_PROTOCOL = freeze.DEFAULT_OUTPUT
OUTPUT_FIELDS = [
    "record_id",
    "event_id",
    "station_key",
    "site_rank",
    "label_source",
    "label_stratum",
    "pick_time_weight",
    "model_key",
    "method_id",
    "comparison_mode",
    "reference_p_time_utc",
    "origin_time_utc",
    "record_start_time_utc",
    "reference_p_seconds",
    "magnitude",
    "epicentral_distance_km",
    "sample_rate_hz",
    "n_samples",
    "response_sha256",
    "orientation_group_sha256",
    "waveform_sha256",
    "edge_filled_component_samples",
    "candidate_count",
    "matched_candidate_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "within_0p5_seconds",
    "within_1p0_seconds",
    "within_1p5_seconds",
    "selected_estimated_p_seconds",
    "pick_error_seconds",
    "absolute_pick_error_seconds",
    "technical_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--model",
        choices=("phasenet_instance", "eqtransformer_original"),
        required=True,
    )
    parser.add_argument("--runner-lease", type=Path, required=True)
    parser.add_argument("--attempt-output-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_protocol(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str, Path]:
    protocol, actual, resolved = freeze.load_bound_json(
        path,
        expected_sha256=expected_sha256,
        expected_protocol_id=freeze.PROTOCOL_ID,
    )
    if protocol.get("status") != (
        "post_hoc_same_record_geonet_2026_public_benchmark_with_"
        "pretuned_weights_thresholds_and_disclosed_edge_gap_remediation"
    ):
        raise RuntimeError("Unexpected final public-baseline status")
    amendment = protocol.get("input_remediation_amendment", {})
    superseded = amendment.get("superseded_v1_protocol", {})
    if (
        amendment.get("prespecification_status")
        != (
            "post_hoc_after_v1_input_validation_failure_before_any_"
            "public_model_result_was_published"
        )
        or not isinstance(superseded, Mapping)
    ):
        raise RuntimeError("Public-baseline input amendment differs")
    freeze.verify_artifact(superseded)
    if int(protocol.get("record_count", -1)) != len(
        protocol.get("record_manifest", [])
    ) or freeze.stable_digest(protocol["record_manifest"]) != protocol.get(
        "record_manifest_sha256"
    ):
        raise RuntimeError("Frozen public-baseline record manifest differs")
    endpoint = protocol.get("endpoint", {})
    remediation = endpoint.get("edge_gap_remediation", {})
    if (
        float(endpoint.get("match_tolerance_seconds", -1.0)) != 1.5
        or endpoint.get("label_stratum") != freeze.PRIMARY_LABEL_STRATUM
        or endpoint.get("availability_latency_generated") is not False
        or endpoint.get("availability_latency_comparison_permitted") is not False
        or endpoint.get("test_set_threshold_tuning_permitted") is not False
        or remediation
        != {
            "maximum_edge_samples_per_side": freeze.EDGE_GAP_MAX_SAMPLES,
            "maximum_nonfinite_component_sample_fraction": (
                freeze.EDGE_GAP_MAX_FRACTION
            ),
            "fill_value": freeze.EDGE_GAP_FILL_VALUE,
            "scope": "first_or_last_one_second_only",
            "role": (
                "post_hoc_input_adapter_correction_after_v1_failed_"
                "before_result_publication"
            ),
            "internal_or_larger_gap_policy": "retain_as_technical_miss",
        }
    ):
        raise RuntimeError("Public-baseline endpoint contract differs")
    execution = protocol.get("execution", {})
    if (
        int(execution.get("workers", -1)) != 1
        or execution.get("serial_models") is not True
        or execution.get("fresh_process_per_model") is not True
        or execution.get("device") != "mps"
        or int(execution.get("torch_threads", -1)) != 1
        or execution.get("runner_lease_id") != runner_lease.LEASE_ID
    ):
        raise RuntimeError("Public-baseline execution contract differs")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Frozen MPS public-baseline device is unavailable")
    allowed_root = Path(execution["allowed_output_root"]).resolve()
    freeze.require_output_path(execution["run_root"], allowed_root=allowed_root)
    freeze.require_output_path(execution["global_lock_path"], allowed_root=allowed_root)
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or freeze.sha256_file(source) != str(expected):
            raise RuntimeError(f"Frozen public-baseline code drifted: {relative}")
    final_path = freeze.verify_artifact(protocol["final_evaluation_protocol"])
    if freeze.sha256_file(final_path) != protocol["final_evaluation_protocol_sha256"]:
        raise RuntimeError("Nested final evaluation protocol differs")
    for spec in protocol["frozen_final_inputs"].values():
        freeze.verify_artifact(spec)
    source_path = freeze.verify_artifact(protocol["source_public_model_protocol"])
    if (
        freeze.sha256_file(source_path)
        != protocol["source_public_model_protocol_sha256"]
    ):
        raise RuntimeError("Nested public-model protocol differs")
    if set(protocol["public_models"]) != freeze.EXPECTED_MODELS:
        raise RuntimeError("Frozen public-model set differs")
    for model_key, spec in protocol["public_models"].items():
        methods = {str(point["id"]) for point in spec["operating_points"]}
        if methods != freeze.EXPECTED_METHODS[model_key]:
            raise RuntimeError(f"Frozen public methods differ: {model_key}")
        if any(
            int(point.get("threshold_count", -1)) != 1
            for point in spec["operating_points"]
        ):
            raise RuntimeError(f"Public threshold count differs: {model_key}")
        for artifact_spec in spec["artifacts"].values():
            freeze.verify_artifact(artifact_spec)
    if sealed_public.software_versions() != protocol["software_versions"]:
        raise RuntimeError("Public-baseline software versions differ")
    return protocol, actual, resolved


def run_root(protocol: Mapping[str, Any], model: str) -> Path:
    allowed = Path(protocol["execution"]["allowed_output_root"]).resolve()
    return (
        freeze.require_output_path(
            protocol["execution"]["run_root"],
            allowed_root=allowed,
        )
        / model
    )


def verify_runner_attempt(
    *,
    lease_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    model: str,
    attempt_root: Path,
) -> dict[str, Any]:
    published = run_root(protocol, model)
    if not lease_path.is_file():
        raise RuntimeError("Evaluator requires a live runner lease")
    return runner_lease.verify_lease(
        lease_path,
        protocol_sha256=protocol_sha256,
        unit={"model": model},
        evaluator_script=Path(__file__).resolve(),
        attempt_root=attempt_root,
        published_root=published,
    )


def orientation_groups(
    protocol: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    path = freeze.verify_artifact(
        protocol["frozen_final_inputs"]["event_station_orientations"]
    )
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        groups[row["group_id"]].append(row)
    for group_id, values in groups.items():
        values.sort(key=lambda item: int(item["component_index"]))
        if len(values) != 3:
            raise RuntimeError(f"Orientation group differs: {group_id}")
    return groups


def waveform_digest(wave: np.ndarray) -> str:
    return legacy.waveform_digest(np.asarray(wave, dtype=np.float32))


def _blank_number() -> str:
    return ""


def score_method_row(
    record: Mapping[str, Any],
    *,
    model: str,
    method_id: str,
    wave: np.ndarray | None,
    candidates: list[dict[str, Any]],
    technical_error: str,
    edge_filled_component_samples: int = 0,
) -> dict[str, Any]:
    origin = UTCDateTime(str(record["origin_time_utc"]))
    start = origin - 60.0
    reference = UTCDateTime(str(record["reference_p_time_utc"]))
    reference_seconds = float(reference - start)
    if not 0.0 <= reference_seconds <= 180.0:
        raise RuntimeError(f"Reference P lies outside record: {record['record_id']}")
    errors = [
        float(candidate["estimated_p_epoch"]) - reference_seconds
        for candidate in candidates
    ]
    matched = [index for index, error in enumerate(errors) if abs(error) <= 1.5]
    selected_index = (
        min(
            matched,
            key=lambda index: (
                abs(errors[index]),
                float(candidates[index]["estimated_p_epoch"]),
                index,
            ),
        )
        if matched
        else None
    )
    hit = int(selected_index is not None and not technical_error)
    selected = candidates[selected_index] if hit else None
    error = errors[selected_index] if hit else None
    candidate_count = len(candidates) if not technical_error else 0
    return {
        "record_id": record["record_id"],
        "event_id": record["event_id"],
        "station_key": record["station_key"],
        "site_rank": int(record["site_rank"]),
        "label_source": record["label_source"],
        "label_stratum": record["label_stratum"],
        "pick_time_weight": float(record["pick_time_weight"]),
        "model_key": model,
        "method_id": method_id,
        "comparison_mode": "released_full_record_offline_annotation",
        "reference_p_time_utc": record["reference_p_time_utc"],
        "origin_time_utc": record["origin_time_utc"],
        "record_start_time_utc": str(start),
        "reference_p_seconds": reference_seconds,
        "magnitude": float(record["magnitude"]),
        "epicentral_distance_km": float(record["epicentral_distance_km"]),
        "sample_rate_hz": 100.0,
        "n_samples": 0 if wave is None else int(wave.shape[1]),
        "response_sha256": record["response_sha256"],
        "orientation_group_sha256": record["orientation_group_sha256"],
        "waveform_sha256": "" if wave is None else waveform_digest(wave),
        "edge_filled_component_samples": edge_filled_component_samples,
        "candidate_count": candidate_count,
        "matched_candidate_count": hit,
        "true_positive": hit,
        "false_positive": candidate_count - hit,
        "false_negative": 1 - hit,
        "within_0p5_seconds": int(hit and abs(float(error)) <= 0.5),
        "within_1p0_seconds": int(hit and abs(float(error)) <= 1.0),
        "within_1p5_seconds": hit,
        "selected_estimated_p_seconds": (
            _blank_number()
            if selected is None
            else float(selected["estimated_p_epoch"])
        ),
        "pick_error_seconds": (_blank_number() if error is None else float(error)),
        "absolute_pick_error_seconds": (
            _blank_number() if error is None else abs(float(error))
        ),
        "technical_error": technical_error,
    }


def write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_FIELDS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    runner_lease.write_json_exclusive(path, payload)


def _maximum_resident_set_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def read_event_source_or_technical_failure(
    response_path: Path,
) -> tuple[Any | None, str]:
    try:
        return read_resilient_mseed(response_path), ""
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256, _ = load_protocol(
        args.protocol,
        expected_sha256=args.expected_protocol_sha256,
    )
    published_root = run_root(protocol, args.model)
    if published_root.exists():
        raise FileExistsError(published_root)
    allowed = Path(protocol["execution"]["allowed_output_root"]).resolve()
    attempt_root = freeze.require_output_path(
        args.attempt_output_root,
        allowed_root=allowed,
    )
    if (
        attempt_root.parent != published_root.parent
        or not attempt_root.name.startswith(f".{published_root.name}.attempt-")
        or not attempt_root.is_dir()
    ):
        raise RuntimeError("Runner attempt directory differs")
    verify_runner_attempt(
        lease_path=args.runner_lease,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        model=args.model,
        attempt_root=attempt_root,
    )
    unexpected = [
        path.name
        for path in attempt_root.iterdir()
        if path.name != "runner_lease.json" and not path.name.startswith("._")
    ]
    if unexpected:
        raise RuntimeError(f"Runner attempt is not empty: {unexpected}")

    torch.set_num_threads(int(protocol["execution"]["torch_threads"]))
    device = torch.device(str(protocol["execution"]["device"]))
    model, loaded_digest = sealed_public.load_sealed_public_model(
        protocol["public_models"][args.model],
        device=device,
    )
    preprocessing = validate_released_annotation_preprocessing(
        model,
        protocol["public_models"][args.model],
    )
    records = list(protocol["record_manifest"])
    groups = orientation_groups(protocol)
    methods = freeze.EXPECTED_METHODS[args.model]
    remediation = protocol["endpoint"]["edge_gap_remediation"]
    rows: list[dict[str, Any]] = []
    edge_filled_record_ids: set[str] = set()
    edge_filled_component_samples_total = 0
    started = time.time()
    current_event = None
    source_result = None
    source = None
    source_error = ""
    try:
        for index, record in enumerate(records, start=1):
            if current_event != record["event_id"]:
                if source_result is not None:
                    del source
                    del source_result
                    gc.collect()
                    if str(device) == "mps":
                        torch.mps.empty_cache()
                response_path = freeze.reject_forbidden_path(record["response_file"])
                if (
                    not response_path.is_file()
                    or response_path.stat().st_size != int(record["response_bytes"])
                    or freeze.sha256_file(response_path) != record["response_sha256"]
                ):
                    raise RuntimeError(
                        f"Frozen event response differs: {record['event_id']}"
                    )
                source_result, source_error = (
                    read_event_source_or_technical_failure(response_path)
                )
                source = (
                    None if source_result is None else source_result.stream
                )
                current_event = record["event_id"]
            wave: np.ndarray | None = None
            edge_filled_component_samples = 0
            error_text = source_error
            scored = {method_id: [] for method_id in methods}
            orientation_rows = groups[record["orientation_group_id"]]
            if (
                freeze.stable_digest(orientation_rows)
                != record["orientation_group_sha256"]
            ):
                raise RuntimeError("Orientation group content differs")
            station = {
                "network": record["network"],
                "station": record["station"],
                "location": record["location"],
                "component_set": record["component_set"],
            }
            orientations = channel_orientations(
                orientation_rows,
                str(record["component_set"]),
            )
            origin = UTCDateTime(str(record["origin_time_utc"]))
            if not error_text:
                try:
                    wave, _ = stream_to_oriented_gappy_zne_array(
                        source,
                        station_target(station),
                        orientations,
                        origin - 60.0,
                        origin + 120.0,
                        target_sample_rate_hz=100.0,
                        samples_per_packet=50,
                    )
                except (ValueError, RuntimeError) as error:
                    error_text = f"{type(error).__name__}: {error}"
            if not error_text:
                if wave is None or wave.shape != (3, 18000):
                    raise RuntimeError(
                        "Public waveform constructor violated the frozen "
                        f"shape contract: {None if wave is None else wave.shape}"
                    )
                try:
                    wave, edge_filled_component_samples = fill_small_edge_gaps(
                        wave,
                        max_edge_samples=int(
                            remediation["maximum_edge_samples_per_side"]
                        ),
                        max_fraction=float(
                            remediation[
                                "maximum_nonfinite_component_sample_fraction"
                            ]
                        ),
                        fill_value=float(remediation["fill_value"]),
                    )
                except ValueError as error:
                    error_text = f"{type(error).__name__}: {error}"
            if not error_text:
                if edge_filled_component_samples:
                    edge_filled_record_ids.add(str(record["record_id"]))
                    edge_filled_component_samples_total += (
                        edge_filled_component_samples
                    )
                scored = score_released_full_record(
                    wave,
                    protocol["public_models"][args.model],
                    model,
                    station=str(record["station"]),
                )
                if set(scored) != methods:
                    raise RuntimeError("Public method output set differs")
            for method_id in sorted(methods):
                rows.append(
                    score_method_row(
                        record,
                        model=args.model,
                        method_id=method_id,
                        wave=wave,
                        candidates=scored[method_id],
                        technical_error=error_text,
                        edge_filled_component_samples=(
                            edge_filled_component_samples
                        ),
                    )
                )
            if index % 25 == 0 or index == len(records):
                print(
                    f"{args.model} [{index}/{len(records)}]",
                    flush=True,
                )
                gc.collect()
                if str(device) == "mps":
                    torch.mps.empty_cache()
    finally:
        if source_result is not None:
            del source
            del source_result

    records_path = attempt_root / "records.csv"
    write_csv_once(records_path, rows)
    final_records_path = published_root / records_path.name
    expected_rows = len(records) * len(methods)
    technical_records = {row["record_id"] for row in rows if row["technical_error"]}
    summary = {
        "summary_id": "cspp_v27_geonet_final_public_baseline_run_v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "final_evaluation_protocol_sha256": protocol[
            "final_evaluation_protocol_sha256"
        ],
        "record_manifest_sha256": protocol["record_manifest_sha256"],
        "model_key": args.model,
        "methods": sorted(methods),
        "records_requested": len(records),
        "records_emitted_per_method": len(records),
        "output_rows": len(rows),
        "expected_output_rows": expected_rows,
        "technical_failure_records": len(technical_records),
        "edge_filled_record_count": len(edge_filled_record_ids),
        "edge_filled_component_samples": (
            edge_filled_component_samples_total
        ),
        "comparison_mode": "released_full_record_offline_annotation",
        "released_annotation_preprocessing": preprocessing,
        "availability_latency_generated": False,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "software_versions": sealed_public.software_versions(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "loaded_state_dict_sha256": loaded_digest,
        "resolved_weight_artifact": protocol["public_models"][args.model]["artifacts"][
            "weights"
        ],
        "resolved_metadata_artifact": protocol["public_models"][args.model][
            "artifacts"
        ]["metadata"],
        "runtime_seconds": time.time() - started,
        "worker_max_resident_set_bytes": _maximum_resident_set_bytes(),
        "output_fields": OUTPUT_FIELDS,
        "records_artifact": runner_lease.artifact_at(
            records_path,
            final_records_path,
        ),
    }
    if len(rows) != expected_rows:
        raise RuntimeError("Public baseline emitted an unexpected row count")
    write_json_once(attempt_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
