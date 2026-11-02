#!/usr/bin/env python3
"""Build resumable three-seed score caches for the sealed GeoNet holdout."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

from obspy import UTCDateTime
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation import (  # noqa: E402
    build_v27_geonet_temporal_score_cache as development,
)
from scripts.evaluation import v27_runner_lease as runner_lease  # noqa: E402
from cspp_v27_reused_core_score_cache import (  # noqa: E402
    load_ensemble,
    release_device_cache,
    sha256_file,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402


WORK_ROOT = Path("workspace").resolve()
DEFAULT_PROTOCOL = (
    WORK_ROOT
    / "protocol_freeze/cspp_v27c_geonet_final_evaluation_protocol_20260729.json"
)
PROTOCOL_ID = "cspp_v27c_geonet_final_evaluation_v1"
EVENT_BEFORE_SECONDS = 60.0
EVENT_AFTER_SECONDS = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--scope", choices=("event", "background"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--execution-audit", type=Path)
    parser.add_argument("--runner-lease", type=Path, required=True)
    parser.add_argument("--attempt-output-root", type=Path, required=True)
    return parser.parse_args()


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"]).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Frozen artifact differs: {path}")
    if path == Path("external_volume") or path.is_relative_to(
        Path("external_volume")
    ):
        raise RuntimeError(f"WD_BLACK is prohibited: {path}")
    return path


def load_protocol(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    protocol_path = path.resolve()
    actual = sha256_file(protocol_path)
    if actual != expected_sha256:
        raise RuntimeError("Final evaluation protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status")
        != (
            "one_operating_point_and_three_predeclared_seeds_frozen_"
            "before_final_score_access"
        )
    ):
        raise RuntimeError("Unexpected final evaluation protocol")
    point = protocol["operating_point"]
    if (
        int(point["count"]) != 1
        or point["threshold_scan_permitted"] is not False
        or point["model_or_threshold_selection_on_final_holdout"] is not False
    ):
        raise RuntimeError("Final protocol permits model or threshold scan")
    execution = protocol["execution"]
    if (
        execution["inference_device"] != "cpu"
        or int(execution["data_loader_workers"]) != 0
        or execution["mps_multi_process_permitted"] is not False
    ):
        raise RuntimeError("Final execution topology differs")
    output_root = Path(execution["score_output_root"]).resolve()
    if not output_root.is_relative_to(
        Path(execution["allowed_output_root"]).resolve()
    ):
        raise RuntimeError("Final score output escapes its allowed root")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != str(expected):
            raise RuntimeError(f"Frozen evaluation code differs: {relative}")
    verify_artifact(protocol["selection_freeze"])
    verify_artifact(protocol["development_protocol"])
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for spec in protocol["models"].values():
        verify_artifact(spec)
    seeds = sorted(int(spec["seed"]) for spec in protocol["models"].values())
    if seeds != [20261101, 20261102, 20261103]:
        raise RuntimeError("Final protocol model seed set differs")
    selected_arm = str(point["selected_arm"])
    model_arms = {str(spec["arm"]) for spec in protocol["models"].values()}
    modes = {
        str(spec["verifier_feature_mode"])
        for spec in protocol["models"].values()
    }
    expected_mode = {"C0": "none", "C1": "current"}[selected_arm]
    if model_arms != {selected_arm} or modes != {expected_mode}:
        raise RuntimeError("Final selected-arm model configuration differs")
    shared_core = protocol.get("shared_core_integrity", {})
    if (
        len(str(shared_core.get("tensor_digest", ""))) != 64
        or shared_core.get("all_other_tensors_byte_identical") is not True
    ):
        raise RuntimeError("Final shared-core integrity contract differs")
    return protocol, actual


def model_specs(protocol: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        key: {
            "path": spec["path"],
            "sha256": spec["sha256"],
            "verifier_feature_mode": spec["verifier_feature_mode"],
        }
        for key, spec in protocol["models"].items()
    }


def verify_checkpoint_metadata(protocol: dict[str, Any]) -> None:
    for key, spec in protocol["models"].items():
        checkpoint = torch.load(
            verify_artifact(spec),
            map_location="cpu",
            weights_only=False,
        )
        training = checkpoint.get("training_config", {})
        model_config = checkpoint.get("model_config", {})
        if (
            checkpoint.get("model_class") != "CSPPCausalPickerV27C"
            or int(checkpoint.get("epoch", -1)) != 3
            or int(training.get("seed", -1))
            != int(spec["seed"])
            or str(model_config.get("verifier_feature_mode"))
            != str(spec["verifier_feature_mode"])
        ):
            raise RuntimeError(
                f"Final checkpoint metadata differs: {key}"
            )


def unique_rows(
    rows: list[dict[str, str]],
    *,
    key,
    label: str,
) -> dict[Any, dict[str, str]]:
    output = {}
    for row in rows:
        identity = key(row)
        if identity in output:
            raise RuntimeError(f"Duplicate {label}: {identity}")
        output[identity] = row
    return output


def start_runner_watchdog(runner_pid: int) -> None:
    if runner_pid <= 1:
        raise ValueError("runner_pid must identify the parent runner")

    def monitor() -> None:
        while True:
            time.sleep(1.0)
            try:
                os.kill(runner_pid, 0)
            except ProcessLookupError:
                os._exit(97)
            except PermissionError:
                continue

    threading.Thread(
        target=monitor,
        name="final-score-runner-watchdog",
        daemon=True,
    ).start()


def verify_runner_attempt(
    *,
    lease_path: Path,
    attempt_root: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = Path(protocol["execution"]["allowed_output_root"]).resolve()
    attempt = attempt_root.resolve()
    if not attempt.is_dir() or not attempt.is_relative_to(allowed):
        raise RuntimeError("Final score attempt root differs")
    published = (
        Path(protocol["execution"]["score_output_root"]).resolve() / scope
    )
    lease = runner_lease.verify_lease(
        lease_path,
        protocol_sha256=protocol_sha256,
        unit={"scope": scope},
        evaluator_script=Path(__file__).resolve(),
        attempt_root=attempt,
        published_root=published,
    )
    identity = {
        "runner_lease_sha256": sha256_file(lease_path.resolve()),
        "lease_nonce": str(lease["nonce"]),
        "runner_pid": int(lease["runner_pid"]),
    }
    return lease, identity


def existing_final_unit_valid(
    path: Path,
    *,
    protocol_sha256: str,
    scope: str,
    unit_key: str,
    input_identity: str,
    run_identity: Mapping[str, Any],
) -> bool:
    if not development.existing_unit_valid(
        path,
        protocol_sha256=protocol_sha256,
        scope=scope,
        unit_key=unit_key,
        input_identity=input_identity,
    ):
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary.get("execution_run_identity") == dict(run_identity)


def event_units(
    protocol: dict[str, Any],
    protocol_sha256: str,
    ensemble: Any,
    *,
    run_identity: Mapping[str, Any],
    limit: int,
    shard_count: int,
    shard_index: int,
) -> None:
    events = development.read_csv(
        verify_artifact(protocol["inputs"]["event_all_events"])
    )
    stations = development.read_csv(
        verify_artifact(protocol["inputs"]["event_scorable_stations"])
    )
    responses = development.read_csv(
        verify_artifact(protocol["inputs"]["event_response_files"])
    )
    orientations = development.orientation_groups(
        development.read_csv(
            verify_artifact(
                protocol["inputs"]["event_station_orientations"]
            )
        )
    )
    event_by_id = unique_rows(
        events,
        key=lambda row: row["event_id"],
        label="final event ID",
    )
    stations_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    station_identities = set()
    for row in stations:
        identity = (row["event_id"], development.station_key(row))
        if identity in station_identities:
            raise RuntimeError(
                f"Duplicate final scorable station: {identity}"
            )
        station_identities.add(identity)
        stations_by_event[row["event_id"]].append(row)
    for values in stations_by_event.values():
        values.sort(key=lambda row: int(row["site_rank"]))
    response_by_event = unique_rows(
        responses,
        key=lambda row: row["event_id"],
        label="final event response",
    )
    if (
        set(stations_by_event) != set(response_by_event)
        or not set(response_by_event).issubset(event_by_id)
    ):
        raise RuntimeError("Scorable event/response identities differ")
    expected_orientation_groups = {
        f"{event_id}.{key}" for event_id, key in station_identities
    }
    if not expected_orientation_groups.issubset(orientations):
        raise RuntimeError("A final scorable station lacks orientation rows")
    ordered_events = sorted(
        (event_by_id[event_id] for event_id in response_by_event),
        key=lambda row: (row["origin_time_utc"], row["event_id"]),
    )

    output_root = Path(protocol["execution"]["score_output_root"])
    status_path = output_root / "event_status.jsonl"
    processed = 0
    for index, event in enumerate(ordered_events, start=1):
        if (index - 1) % shard_count != shard_index:
            continue
        if limit and processed >= limit:
            break
        event_id = event["event_id"]
        response = response_by_event[event_id]
        input_identity = development.stable_hash(
            {
                "event": event,
                "stations": stations_by_event[event_id],
                "response_sha256": response["response_sha256"],
                "orientation_sha256": protocol["inputs"][
                    "event_station_orientations"
                ]["sha256"],
            }
        )
        unit_root = output_root / "event" / event_id
        summary_path = unit_root / "summary.json"
        if existing_final_unit_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            scope="event",
            unit_key=event_id,
            input_identity=input_identity,
            run_identity=run_identity,
        ):
            print(
                f"event {index}/{len(ordered_events)} {event_id}: cached",
                flush=True,
            )
            continue
        development.append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_started",
                "scope": "event",
                "unit_key": event_id,
            },
        )
        started = time.perf_counter()
        source_path = Path(response["response_file"]).resolve()
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(response["response_bytes"])
            or sha256_file(source_path) != response["response_sha256"]
        ):
            raise RuntimeError(f"Final event waveform differs: {source_path}")
        source_result = read_resilient_mseed(source_path)
        source = source_result.stream
        origin = UTCDateTime(event["origin_time_utc"])
        start = origin - EVENT_BEFORE_SECONDS
        end = origin + EVENT_AFTER_SECONDS
        station_results = []
        artifacts = {}
        for station in stations_by_event[event_id]:
            key = development.station_key(station)
            group_id = f"{event_id}.{key}"
            arrays, metadata, local_snr = development.score_station(
                source=source,
                station=station,
                orientation_rows=orientations[group_id],
                start=start,
                end=end,
                ensemble=ensemble,
                protocol=protocol,
                reference_p_epoch=float(
                    UTCDateTime(station["p_arrival_time_utc"]).timestamp
                ),
            )
            cache_path = unit_root / f"{key}.npz"
            development.write_npz_atomic(cache_path, arrays)
            artifacts[key] = development.artifact(cache_path)
            station_results.append(
                {
                    "station_key": key,
                    "site_rank": int(station["site_rank"]),
                    "label_source": station["label_source"],
                    "pick_time_weight": (
                        float(station["pick_time_weight"])
                        if station.get("pick_time_weight")
                        else None
                    ),
                    "reference_p_epoch": float(
                        UTCDateTime(
                            station["p_arrival_time_utc"]
                        ).timestamp
                    ),
                    "epicentral_distance_km": float(
                        station["epicentral_distance_km"]
                    ),
                    "magnitude": float(station["magnitude"]),
                    "local_p_snr": local_snr,
                    "metadata": metadata,
                    "score_cache": artifacts[key],
                }
            )
            release_device_cache(ensemble.device_name)
        decode_diagnostics = source_result.diagnostics
        del source
        del source_result
        summary = {
            "protocol_sha256": protocol_sha256,
            "scope": "event",
            "unit_key": event_id,
            "input_identity": input_identity,
            "status": "complete",
            "execution_run_identity": dict(run_identity),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "origin_time_utc": event["origin_time_utc"],
            "magnitude": float(event["magnitude"]),
            "station_count": len(station_results),
            "station_results": station_results,
            "event_waveform_decode": decode_diagnostics,
            "shared_core_digest": ensemble.shared_core_digest,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": (
                development.maximum_resident_set_bytes()
            ),
            "artifacts": artifacts,
        }
        development.write_json_atomic(summary_path, summary)
        development.append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_completed",
                "scope": "event",
                "unit_key": event_id,
                "summary": development.artifact(summary_path),
            },
        )
        processed += 1
        print(
            f"event {index}/{len(ordered_events)} {event_id}: "
            f"{len(station_results)} stations, "
            f"{summary['elapsed_seconds']:.2f}s",
            flush=True,
        )


def background_units(
    protocol: dict[str, Any],
    protocol_sha256: str,
    ensemble: Any,
    *,
    run_identity: Mapping[str, Any],
    limit: int,
    shard_count: int,
    shard_index: int,
) -> None:
    dates = development.read_csv(
        verify_artifact(protocol["inputs"]["background_dates"])
    )
    stations = development.read_csv(
        verify_artifact(protocol["inputs"]["background_station_days"])
    )
    objects = development.read_csv(
        verify_artifact(protocol["inputs"]["background_objects"])
    )
    orientations = development.orientation_groups(
        development.read_csv(
            verify_artifact(
                protocol["inputs"]["background_station_orientations"]
            )
        )
    )
    stations_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    objects_by_station: dict[
        tuple[str, str], list[dict[str, str]]
    ] = defaultdict(list)
    date_by_key = unique_rows(
        dates,
        key=lambda row: row["date_utc"],
        label="final background date",
    )
    station_identities = set()
    for row in stations:
        identity = (row["date_utc"], development.station_key(row))
        if identity in station_identities:
            raise RuntimeError(
                f"Duplicate final background station-day: {identity}"
            )
        station_identities.add(identity)
        stations_by_date[row["date_utc"]].append(row)
    object_identities = set()
    for row in objects:
        station_identity = (
            row["date_utc"],
            development.station_key(row),
        )
        identity = (*station_identity, row["channel"])
        if identity in object_identities:
            raise RuntimeError(
                f"Duplicate final background object: {identity}"
            )
        object_identities.add(identity)
        objects_by_station[station_identity].append(row)
    if set(stations_by_date) != set(date_by_key):
        raise RuntimeError("Final background date/station identities differ")
    if any(len(values) != 6 for values in stations_by_date.values()):
        raise RuntimeError(
            "Every final background date must contain six unique stations"
        )
    if set(objects_by_station) != station_identities:
        raise RuntimeError("Final background station/object identities differ")
    for identity, values in objects_by_station.items():
        if (
            len(values) != 3
            or len({row["channel"] for row in values}) != 3
        ):
            raise RuntimeError(
                f"Final background component set differs: {identity}"
            )
    physical_stations = {
        identity[1] for identity in station_identities
    }
    if set(orientations) != physical_stations:
        raise RuntimeError(
            "Final background orientation inventory differs"
        )
    for values in stations_by_date.values():
        values.sort(key=development.station_key)
    for values in objects_by_station.values():
        values.sort(key=lambda row: row["channel"])
    dates.sort(key=lambda row: int(row["quiet_rank"]))

    output_root = Path(protocol["execution"]["score_output_root"])
    status_path = output_root / "background_status.jsonl"
    processed = 0
    for index, date in enumerate(dates, start=1):
        if (index - 1) % shard_count != shard_index:
            continue
        if limit and processed >= limit:
            break
        date_utc = date["date_utc"]
        input_objects = []
        for station in stations_by_date[date_utc]:
            input_objects.extend(
                {
                    "station_key": development.station_key(station),
                    "channel": row["channel"],
                    "local_sha256": row["local_sha256"],
                    "source_version_id": row["source_version_id"],
                }
                for row in objects_by_station[
                    (date_utc, development.station_key(station))
                ]
            )
        input_identity = development.stable_hash(
            {
                "date": date,
                "stations": stations_by_date[date_utc],
                "objects": input_objects,
                "orientation_sha256": protocol["inputs"][
                    "background_station_orientations"
                ]["sha256"],
            }
        )
        unit_root = output_root / "background" / date_utc
        summary_path = unit_root / "summary.json"
        if existing_final_unit_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            scope="background",
            unit_key=date_utc,
            input_identity=input_identity,
            run_identity=run_identity,
        ):
            print(
                f"background {index}/{len(dates)} {date_utc}: cached",
                flush=True,
            )
            continue
        development.append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_started",
                "scope": "background",
                "unit_key": date_utc,
            },
        )
        started = time.perf_counter()
        start = UTCDateTime(f"{date_utc}T00:00:00Z")
        end = start + 86400.0
        station_results = []
        artifacts = {}
        for station in stations_by_date[date_utc]:
            key = development.station_key(station)
            source_rows = objects_by_station[(date_utc, key)]
            if len(source_rows) != 3:
                raise RuntimeError(
                    f"Expected three background objects: {date_utc}/{key}"
                )
            source_paths = []
            for row in source_rows:
                path = Path(row["local_path"]).resolve()
                if (
                    not path.is_file()
                    or path.stat().st_size != int(row["local_bytes"])
                    or sha256_file(path) != row["local_sha256"]
                ):
                    raise RuntimeError(
                        f"Final background waveform differs: {path}"
                    )
                source_paths.append(path)
            arrays, metadata, _ = development.score_station(
                source=source_paths,
                station=station,
                orientation_rows=orientations[key],
                start=start,
                end=end,
                ensemble=ensemble,
                protocol=protocol,
                reference_p_epoch=None,
            )
            cache_path = unit_root / f"{key}.npz"
            development.write_npz_atomic(cache_path, arrays)
            artifacts[key] = development.artifact(cache_path)
            availability = float(
                station["post_quarantine_common_availability"]
            )
            if availability < 0.95:
                raise RuntimeError(
                    f"Background availability differs: {date_utc}/{key}"
                )
            station_results.append(
                {
                    "station_key": key,
                    "common_availability": availability,
                    "metadata": metadata,
                    "score_cache": artifacts[key],
                    "source_objects": [
                        {
                            "channel": row["channel"],
                            "local_sha256": row["local_sha256"],
                            "source_version_id": row["source_version_id"],
                            "quarantined_record_count": int(
                                row["quarantined_record_count"]
                            ),
                        }
                        for row in source_rows
                    ],
                }
            )
            release_device_cache(ensemble.device_name)
        summary = {
            "protocol_sha256": protocol_sha256,
            "scope": "background",
            "unit_key": date_utc,
            "input_identity": input_identity,
            "status": "complete",
            "execution_run_identity": dict(run_identity),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "quiet_rank": int(date["quiet_rank"]),
            "station_count": len(station_results),
            "station_results": station_results,
            "post_quarantine_station_exposure_days": float(
                date["post_quarantine_station_exposure_days"]
            ),
            "post_quarantine_connected_network_exposure_days": float(
                date[
                    "post_quarantine_connected_network_exposure_days"
                ]
            ),
            "post_quarantine_minimum_three_station_exposure_days": float(
                date[
                    "post_quarantine_minimum_three_station_exposure_days"
                ]
            ),
            "shared_core_digest": ensemble.shared_core_digest,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": (
                development.maximum_resident_set_bytes()
            ),
            "artifacts": artifacts,
        }
        development.write_json_atomic(summary_path, summary)
        development.append_jsonl(
            status_path,
            {
                "time_utc": datetime.now(UTC).isoformat(),
                "event": "unit_completed",
                "scope": "background",
                "unit_key": date_utc,
                "summary": development.artifact(summary_path),
            },
        )
        processed += 1
        print(
            f"background {index}/{len(dates)} {date_utc}: "
            f"{len(station_results)} stations, "
            f"{summary['elapsed_seconds']:.2f}s",
            flush=True,
        )


def _command_options(command: list[Any]) -> dict[str, str]:
    if len(command) < 4 or len(command[2:]) % 2:
        raise RuntimeError("Final worker command structure differs")
    options = {}
    for flag, value in zip(command[2::2], command[3::2], strict=True):
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise RuntimeError("Final worker command flag differs")
        if flag in options:
            raise RuntimeError("Final worker command repeats a flag")
        options[flag] = str(value)
    return options


def verify_final_execution_audit(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    scope: str,
    path: str | Path,
) -> dict[str, Any]:
    audit_path = Path(path).resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    execution = protocol["execution"]
    workers = int(execution["inference_processes"])
    threads = int(execution["torch_threads_per_process"])
    interop_threads = int(
        execution["torch_interop_threads_per_process"]
    )
    maximum_combined = round(
        float(execution["combined_worker_rss_limit_gib"]) * 1024**3
    )
    maximum_worker = round(
        float(execution["per_worker_rss_limit_gib"]) * 1024**3
    )
    minimum_available = round(
        float(execution["minimum_system_available_memory_gib"])
        * 1024**3
    )
    maximum_swap = round(
        float(execution["maximum_system_swap_used_gib"]) * 1024**3
    )
    expected = {
        "run_id": "cspp_v27c_geonet_final_bounded_cpu_workers_v2",
        "protocol_sha256": protocol_sha256,
        "scope": scope,
        "status": "workers_completed_within_limits",
        "workers": workers,
        "threads_per_worker": threads,
        "interop_threads_per_worker": interop_threads,
        "maximum_combined_worker_rss_bytes": maximum_combined,
        "maximum_per_worker_rss_bytes": maximum_worker,
        "minimum_system_available_memory_bytes": minimum_available,
        "maximum_system_swap_used_bytes": maximum_swap,
        "rss_poll_interval_seconds": float(
            execution["rss_poll_interval_seconds"]
        ),
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise RuntimeError(f"Final execution audit differs: {key}")
    runner_pid = int(audit.get("runner_pid", -1))
    resource_samples = int(audit.get("resource_sample_count", 0))
    return_codes = audit.get("return_codes", [])
    worker_peaks = audit.get(
        "observed_peak_per_worker_rss_bytes", []
    )
    commands = audit.get("commands", [])
    observed_combined = int(
        audit.get("observed_peak_combined_worker_rss_bytes", -1)
    )
    observed_available = int(
        audit.get(
            "observed_minimum_system_available_memory_bytes", -1
        )
    )
    observed_swap = int(
        audit.get("observed_maximum_system_swap_used_bytes", -1)
    )
    if (
        runner_pid <= 0
        or resource_samples < 1
        or len(return_codes) != workers
        or any(int(code) != 0 for code in return_codes)
        or len(worker_peaks) != workers
        or any(
            int(value) < 0 or int(value) > maximum_worker
            for value in worker_peaks
        )
        or observed_combined < 0
        or observed_combined > maximum_combined
        or observed_combined > sum(int(value) for value in worker_peaks)
        or observed_available < minimum_available
        or observed_swap < 0
        or observed_swap > maximum_swap
        or len(commands) != workers
    ):
        raise RuntimeError("Final execution resource audit differs")

    expected_builder = (
        ROOT / "scripts/evaluation/build_v27_geonet_final_score_cache.py"
    ).resolve()
    expected_runner = (
        ROOT / "scripts/evaluation/run_v27_geonet_final_score_workers.py"
    ).resolve()
    expected_python = Path(
        "external_home/miniforge3/envs/zhy/bin/python"
    ).resolve()
    lease_path = verify_artifact(audit.get("runner_lease", {}))
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    attempt_root = Path(str(lease.get("attempt_root", ""))).resolve()
    published_root = Path(str(lease.get("published_root", ""))).resolve()
    expected_published = (
        Path(execution["score_output_root"]).resolve() / scope
    )
    expected_run_identity = {
        "runner_lease_sha256": sha256_file(lease_path),
        "lease_nonce": str(lease.get("nonce", "")),
        "runner_pid": runner_pid,
    }
    if (
        lease_path != attempt_root / "runner_lease.json"
        or lease.get("lease_id") != runner_lease.LEASE_ID
        or lease.get("protocol_sha256") != protocol_sha256
        or lease.get("unit") != {"scope": scope}
        or int(lease.get("runner_pid", -1)) != runner_pid
        or Path(str(lease.get("runner_script", ""))).resolve()
        != expected_runner
        or lease.get("runner_script_sha256")
        != sha256_file(expected_runner)
        or Path(str(lease.get("evaluator_script", ""))).resolve()
        != expected_builder
        or lease.get("evaluator_script_sha256")
        != sha256_file(expected_builder)
        or published_root != expected_published
        or audit.get("run_identity") != expected_run_identity
    ):
        raise RuntimeError("Final execution runner lease differs")
    observed_shards = set()
    observed_pids = set()
    for entry in commands:
        shard_index = int(entry.get("shard_index", -1))
        pid = int(entry.get("pid", -1))
        command = entry.get("command")
        if (
            not isinstance(command, list)
            or len(command) < 2
            or Path(str(command[0])).resolve() != expected_python
            or Path(str(command[1])).resolve() != expected_builder
            or shard_index < 0
            or shard_index >= workers
            or pid <= 0
        ):
            raise RuntimeError("Final execution command identity differs")
        options = _command_options(command)
        expected_options = {
            "--expected-protocol-sha256": protocol_sha256,
            "--scope": scope,
            "--device": "cpu",
            "--shard-count": str(workers),
            "--shard-index": str(shard_index),
            "--runner-lease": str(lease_path),
            "--attempt-output-root": str(attempt_root),
        }
        for key, value in expected_options.items():
            if options.get(key) != value:
                raise RuntimeError(
                    f"Final execution command differs: {key}"
                )
        protocol_path = Path(options.get("--protocol", "")).resolve()
        if (
            set(options) != {"--protocol", *expected_options}
            or not protocol_path.is_file()
            or sha256_file(protocol_path) != protocol_sha256
        ):
            raise RuntimeError("Final execution protocol command differs")
        observed_shards.add(shard_index)
        observed_pids.add(pid)
    if observed_shards != set(range(workers)) or len(observed_pids) != workers:
        raise RuntimeError("Final execution shard or PID set differs")
    return audit


def write_scope_completion(
    protocol: dict[str, Any],
    protocol_sha256: str,
    scope: str,
    *,
    execution_audit_path: Path,
) -> None:
    output_root = Path(protocol["execution"]["score_output_root"])
    expected = int(
        protocol["expected_units"][
            "event_waveform_files"
            if scope == "event"
            else "background_scoring_dates"
        ]
    )
    unit_root = output_root / scope
    summaries = (
        sorted(unit_root.glob("*/summary.json"))
        if unit_root.is_dir()
        else []
    )
    if len(summaries) != expected:
        return
    if scope == "event":
        expected_unit_keys = {
            row["event_id"]
            for row in development.read_csv(
                verify_artifact(protocol["inputs"]["event_response_files"])
            )
        }
        expected_stations_by_unit: dict[str, set[str]] = defaultdict(set)
        for row in development.read_csv(
            verify_artifact(
                protocol["inputs"]["event_scorable_stations"]
            )
        ):
            expected_stations_by_unit[row["event_id"]].add(
                development.station_key(row)
            )
    else:
        expected_unit_keys = {
            row["date_utc"]
            for row in development.read_csv(
                verify_artifact(protocol["inputs"]["background_dates"])
            )
        }
        expected_stations_by_unit = defaultdict(set)
        for row in development.read_csv(
            verify_artifact(
                protocol["inputs"]["background_station_days"]
            )
        ):
            expected_stations_by_unit[row["date_utc"]].add(
                development.station_key(row)
            )
    if len(expected_unit_keys) != expected:
        raise RuntimeError(f"Frozen final {scope} unit identities differ")
    unit_specs = []
    observed_unit_keys = set()
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256") != protocol_sha256
            or summary.get("scope") != scope
            or summary.get("status") != "complete"
        ):
            raise RuntimeError(f"Final unit summary differs: {path}")
        unit_key = str(summary["unit_key"])
        if path.parent.name != unit_key or unit_key in observed_unit_keys:
            raise RuntimeError(
                f"Final unit summary path/identity differs: {path}"
            )
        observed_unit_keys.add(unit_key)
        station_results = summary.get("station_results", [])
        observed_stations = [
            str(row["station_key"]) for row in station_results
        ]
        if (
            len(observed_stations) != len(set(observed_stations))
            or set(observed_stations)
            != expected_stations_by_unit[unit_key]
            or int(summary.get("station_count", -1))
            != len(observed_stations)
            or set(summary.get("artifacts", {}))
            != set(observed_stations)
        ):
            raise RuntimeError(
                f"Final unit station identities differ: {path}"
            )
        if summary.get("shared_core_digest") != protocol[
            "shared_core_integrity"
        ]["tensor_digest"]:
            raise RuntimeError(
                f"Final unit shared-core digest differs: {path}"
            )
        for spec in summary["artifacts"].values():
            verify_artifact(spec)
        unit_specs.append(development.artifact(path))
    if observed_unit_keys != expected_unit_keys:
        raise RuntimeError(f"Final {scope} unit set differs")
    execution_audit = execution_audit_path.resolve()
    if not execution_audit.is_file():
        raise FileNotFoundError(execution_audit)
    execution_audit_payload = verify_final_execution_audit(
        protocol,
        protocol_sha256,
        scope,
        execution_audit,
    )
    run_identity = execution_audit_payload["run_identity"]
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("execution_run_identity") != run_identity:
            raise RuntimeError(
                f"Final unit was produced outside this audited run: {path}"
            )
    execution_audit_spec = development.artifact(execution_audit)
    completion_path = output_root / f"{scope}_score_cache_complete.json"
    if completion_path.exists():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            existing.get("completion_id")
            != f"cspp_v27c_geonet_final_{scope}_score_cache_v1"
            or existing.get("protocol_sha256") != protocol_sha256
            or int(existing.get("units", -1)) != len(unit_specs)
            or existing.get("execution_run_identity") != run_identity
            or existing.get("unit_summaries") != unit_specs
            or existing.get("execution_audit") != execution_audit_spec
        ):
            raise RuntimeError(
                f"Existing final completion differs: {completion_path}"
            )
        return
    completion = {
        "completion_id": f"cspp_v27c_geonet_final_{scope}_score_cache_v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "scope": scope,
        "units": len(unit_specs),
        "inference_device": protocol["execution"]["inference_device"],
        "inference_processes": int(
            protocol["execution"]["inference_processes"]
        ),
        "data_loader_workers": 0,
        "shared_core_digest": protocol["shared_core_integrity"][
            "tensor_digest"
        ],
        "execution_run_identity": run_identity,
        "execution_audit": execution_audit_spec,
        "unit_summaries": unit_specs,
    }
    development.write_json_atomic(completion_path, completion)


def main() -> None:
    args = parse_args()
    if (
        args.shard_count < 1
        or args.shard_index < 0
        or args.shard_index >= args.shard_count
    ):
        raise ValueError("Invalid shard topology")
    protocol, protocol_sha256 = load_protocol(
        args.protocol, args.expected_protocol_sha256
    )
    if args.device != protocol["execution"]["inference_device"]:
        raise RuntimeError("Requested device differs from frozen protocol")
    if args.shard_count != int(
        protocol["execution"]["inference_processes"]
    ):
        raise RuntimeError("Requested shard count differs from protocol")
    lease, run_identity = verify_runner_attempt(
        lease_path=args.runner_lease,
        attempt_root=args.attempt_output_root,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        scope=args.scope,
    )
    if args.finalize_only:
        if args.execution_audit is None:
            raise RuntimeError(
                "--finalize-only requires --execution-audit"
            )
        write_scope_completion(
            protocol,
            protocol_sha256,
            args.scope,
            execution_audit_path=args.execution_audit,
        )
        return
    start_runner_watchdog(int(lease["runner_pid"]))
    torch.set_num_interop_threads(
        int(protocol["execution"]["torch_interop_threads_per_process"])
    )
    verify_checkpoint_metadata(protocol)
    ensemble = load_ensemble(
        model_specs(protocol),
        device_name=args.device,
        torch_threads=int(
            protocol["execution"]["torch_threads_per_process"]
        ),
    )
    if ensemble.shared_core_digest != protocol[
        "shared_core_integrity"
    ]["tensor_digest"]:
        raise RuntimeError("Loaded final shared-core digest differs")
    if set(ensemble.event_heads) != set(protocol["models"]):
        raise RuntimeError("Loaded final event-head identities differ")
    if args.scope == "event":
        event_units(
            protocol,
            protocol_sha256,
            ensemble,
            run_identity=run_identity,
            limit=args.limit,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
    else:
        background_units(
            protocol,
            protocol_sha256,
            ensemble,
            run_identity=run_identity,
            limit=args.limit,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )


if __name__ == "__main__":
    main()
