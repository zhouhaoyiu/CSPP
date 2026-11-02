#!/usr/bin/env python3
"""Build memory-bounded CSPP score caches for frozen NCSN station-days."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
from obspy import UTCDateTime


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cspp_v27_reused_core_score_cache import (  # noqa: E402
    load_ensemble,
    release_device_cache,
    score_waveform_packets,
    sha256_file,
)
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    ChannelOrientation,
    stream_to_oriented_gappy_zne_array,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_continuous_scoring_protocol_20260817.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def maximum_resident_set_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def station_day_id(row: dict[str, str]) -> str:
    location = row["location"] or "--"
    return f"{row['date_utc']}__{row['network']}.{row['station']}.{location}"


def station_key(row: dict[str, str]) -> str:
    return ".".join((row["network"], row["station"], row["location"]))


def orientation_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    for group_id, values in groups.items():
        values.sort(key=lambda row: int(row["component_index"]))
        if len(values) != 3:
            raise RuntimeError(f"Orientation group is incomplete: {group_id}")
    return groups


def load_protocol(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Scoring protocol SHA-256 differs: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_ncedc_continuous_scoring_v1":
        raise RuntimeError("Unexpected NCSN continuous scoring protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Scoring code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for spec in protocol["models"].values():
        verify_artifact(spec)
    return protocol, actual


def unit_root(output_root: Path, row: dict[str, str]) -> Path:
    location = row["location"] or "--"
    return output_root / "station_day" / row["date_utc"] / (
        f"{row['network']}.{row['station']}.{location}"
    )


def summary_valid(
    path: Path,
    *,
    protocol_sha256: str,
    identity: str,
    input_identity: str,
) -> bool:
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": protocol_sha256,
        "station_day_id": identity,
        "input_identity": input_identity,
        "status": "complete",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Existing summary differs in {key}: {path}")
    verify_artifact(summary["score_cache"])
    return True


def score_unit(
    *,
    request: dict[str, str],
    acquisition: dict[str, str],
    orientation_rows: list[dict[str, str]],
    ensemble: Any,
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    source_path = Path(acquisition["mseed_file"])
    source_result = read_resilient_mseed(source_path)
    start = UTCDateTime(request["request_start_utc"])
    end = UTCDateTime(request["request_end_utc"])
    orientations = [
        ChannelOrientation(
            network=row["network"],
            station=row["station"],
            location=row["location"],
            channel=row["channel"],
            azimuth=float(row["azimuth_deg"]),
            dip=float(row["dip_deg"]),
            scale=float(row["scale"]),
            scale_units=row["scale_units"],
        )
        for row in orientation_rows
    ]
    wave, stream_metadata = stream_to_oriented_gappy_zne_array(
        source_result.stream,
        {
            "network": request["network"],
            "station": request["station"],
            "location": request["location"],
            "latitude": float(request["station_latitude"]),
            "longitude": float(request["station_longitude"]),
        },
        orientations,
        start,
        end,
        target_sample_rate_hz=float(protocol["runtime"]["sample_rate_hz"]),
        samples_per_packet=int(protocol["runtime"]["packet_samples"]),
    )
    cache = score_waveform_packets(
        wave,
        start_epoch=float(start.timestamp),
        ensemble=ensemble,
        runtime=protocol["runtime"],
    )
    arrays = cache.arrays()
    packet_seconds = float(protocol["runtime"]["packet_duration_seconds"])
    duration = {
        "total_packets": int(len(cache.quality_valid)),
        "quality_valid_packets": int(cache.quality_valid.sum()),
        "candidate_allowed_packets": int(cache.candidate_allowed.sum()),
        "quality_valid_station_days": float(
            cache.quality_valid.sum() * packet_seconds / 86400.0
        ),
        "candidate_eligible_station_days": float(
            cache.candidate_allowed.sum() * packet_seconds / 86400.0
        ),
    }
    metadata = {
        "stream": stream_metadata,
        "decode": source_result.diagnostics,
        "score": cache.diagnostics,
        "start_epoch": float(start.timestamp),
        "end_epoch": float(end.timestamp),
    }
    del cache
    del wave
    del source_result
    return arrays, metadata, duration


def finalize(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    requests: list[dict[str, str]],
    acquisitions: dict[str, dict[str, str]],
) -> None:
    output_root = Path(protocol["execution"]["output_root"])
    summary_specs: list[dict[str, Any]] = []
    quality_days = 0.0
    eligible_days = 0.0
    for request in requests:
        identity = station_day_id(request)
        acquisition = acquisitions[identity]
        if acquisition["status"] == "download_or_parse_error":
            continue
        path = unit_root(output_root, request) / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"Missing station-day score summary: {identity}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_sha256") != protocol_sha256
            or summary.get("station_day_id") != identity
            or summary.get("status") != "complete"
        ):
            raise RuntimeError(f"Station-day summary differs: {path}")
        verify_artifact(summary["score_cache"])
        quality_days += float(summary["duration"]["quality_valid_station_days"])
        eligible_days += float(
            summary["duration"]["candidate_eligible_station_days"]
        )
        summary_specs.append(artifact(path))
    expected = int(protocol["expected_units"]["data_bearing_station_days"])
    if len(summary_specs) != expected:
        raise RuntimeError(
            f"Expected {expected} station-day summaries, got {len(summary_specs)}"
        )
    completion = {
        "completion_id": "cspp_v27_ncedc_continuous_score_cache_v1",
        "protocol_sha256": protocol_sha256,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "data_bearing_station_days": len(summary_specs),
        "archive_unavailable_station_days": int(
            protocol["expected_units"]["archive_unavailable_station_days"]
        ),
        "quality_valid_station_days": quality_days,
        "candidate_eligible_station_days": eligible_days,
        "unit_summaries": summary_specs,
        "model_outputs_computed": True,
    }
    write_json_atomic(output_root / "continuous_score_cache_complete.json", completion)
    print(json.dumps(completion | {"unit_summaries": "omitted"}, indent=2))


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    protocol, protocol_sha256 = load_protocol(
        args.protocol.resolve(), args.expected_protocol_sha256
    )
    requests = read_csv(verify_artifact(protocol["inputs"]["station_day_requests"]))
    requests.sort(key=lambda row: (row["date_utc"], int(row["availability_rank"])))
    acquisition_rows = read_csv(
        verify_artifact(protocol["inputs"]["acquisition_status"])
    )
    acquisitions = {row["station_day_id"]: row for row in acquisition_rows}
    if len(acquisitions) != len(requests):
        raise RuntimeError("Acquisition and request counts differ")
    if args.finalize_only:
        finalize(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            requests=requests,
            acquisitions=acquisitions,
        )
        return

    orientations = orientation_groups(
        read_csv(verify_artifact(protocol["inputs"]["station_day_orientations"]))
    )
    ensemble = load_ensemble(
        {
            key: {
                "path": spec["path"],
                "sha256": spec["sha256"],
                "verifier_feature_mode": spec["verifier_feature_mode"],
            }
            for key, spec in protocol["models"].items()
        },
        device_name=args.device,
        torch_threads=int(protocol["execution"]["torch_threads_per_process"]),
    )
    output_root = Path(protocol["execution"]["output_root"])
    status_path = output_root / f"status_shard_{args.shard_index:02d}.jsonl"
    processed = 0
    for index, request in enumerate(requests):
        if index % args.shard_count != args.shard_index:
            continue
        acquisition = acquisitions[station_day_id(request)]
        if acquisition["status"] == "download_or_parse_error":
            continue
        if args.limit and processed >= args.limit:
            break
        identity = station_day_id(request)
        source_path = Path(acquisition["mseed_file"])
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(acquisition["mseed_bytes"])
            or sha256_file(source_path) != acquisition["mseed_sha256"]
        ):
            raise RuntimeError(f"Continuous waveform drifted: {source_path}")
        qc_path = Path(acquisition["qc_file"])
        input_identity = stable_hash(
            {
                "request": request,
                "acquisition": acquisition,
                "qc_sha256": sha256_file(qc_path),
                "orientation_sha256": protocol["inputs"][
                    "station_day_orientations"
                ]["sha256"],
            }
        )
        root = unit_root(output_root, request)
        summary_path = root / "summary.json"
        if summary_valid(
            summary_path,
            protocol_sha256=protocol_sha256,
            identity=identity,
            input_identity=input_identity,
        ):
            print(f"[{index + 1}/{len(requests)}] {identity}: cached", flush=True)
            continue
        started = time.perf_counter()
        arrays, metadata, duration = score_unit(
            request=request,
            acquisition=acquisition,
            orientation_rows=orientations[identity],
            ensemble=ensemble,
            protocol=protocol,
        )
        cache_path = root / "scores.npz"
        write_npz_atomic(cache_path, arrays)
        score_spec = artifact(cache_path)
        summary = {
            "protocol_sha256": protocol_sha256,
            "station_day_id": identity,
            "date_utc": request["date_utc"],
            "station_key": station_key(request),
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "acquisition_status": acquisition["status"],
            "acquisition_minimum_component_available_fraction": float(
                acquisition["minimum_component_available_fraction"]
            ),
            "duration": duration,
            "metadata": metadata,
            "score_cache": score_spec,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": maximum_resident_set_bytes(),
        }
        write_json_atomic(summary_path, summary)
        with status_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "completed_at_utc": summary["completed_at_utc"],
                        "station_day_id": identity,
                        "summary": artifact(summary_path),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"[{index + 1}/{len(requests)}] {identity}: "
            f"{duration['candidate_eligible_station_days']:.4f} eligible days, "
            f"{summary['elapsed_seconds']:.1f}s, "
            f"rss={summary['maximum_resident_set_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )
        processed += 1
        del arrays
        del metadata
        gc.collect()
        release_device_cache(ensemble.device_name)


if __name__ == "__main__":
    main()
