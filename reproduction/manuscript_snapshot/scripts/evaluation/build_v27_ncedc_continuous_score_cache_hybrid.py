#!/usr/bin/env python3
"""Build NCSN continuous score caches with the frozen hybrid implementation."""
from __future__ import annotations

import argparse
import gc
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from obspy import UTCDateTime
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

import build_v27_ncedc_continuous_score_cache as legacy  # noqa: E402
from cspp_v27_hybrid_score_cache import (  # noqa: E402
    prepare_hybrid_ensemble,
    release_hybrid_cache,
    score_waveform_packets_hybrid,
)
from cspp_v27_reused_core_score_cache import load_ensemble, sha256_file  # noqa: E402
from oriented_gappy_strong_motion_stream import (  # noqa: E402
    ChannelOrientation,
    stream_to_oriented_gappy_zne_array,
)
from resilient_mseed import read_resilient_mseed  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_continuous_scoring_hybrid_amendment_20260817.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--equivalence-report", type=Path)
    parser.add_argument("--expected-equivalence-sha256")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def load_protocol(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Hybrid amendment SHA-256 differs: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "cspp_v27_ncedc_continuous_hybrid_v2":
        raise RuntimeError("Unexpected NCSN hybrid scoring protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"Hybrid scoring code drifted: {relative}")
    for spec in protocol["inputs"].values():
        legacy.verify_artifact(spec)
    for spec in protocol["models"].values():
        legacy.verify_artifact(spec)
    return protocol, actual


def verify_equivalence(
    path: Path | None,
    expected_sha256: str | None,
    protocol_sha256: str,
) -> None:
    if path is None or expected_sha256 is None:
        raise RuntimeError("Full scoring requires the frozen equivalence report")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"Equivalence report SHA-256 differs: {actual}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("decision") != "pass"
        or report.get("hybrid_amendment_sha256") != protocol_sha256
    ):
        raise RuntimeError("Hybrid equivalence gate did not pass")


def score_unit(
    *,
    request: dict[str, str],
    acquisition: dict[str, str],
    orientation_rows: list[dict[str, str]],
    ensemble: Any,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_result = read_resilient_mseed(Path(acquisition["mseed_file"]))
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
    cache = score_waveform_packets_hybrid(
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
    del cache, wave, source_result
    return arrays, metadata, duration


def mps_memory() -> dict[str, int]:
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(
        args.protocol.resolve(), args.expected_protocol_sha256
    )
    if not args.gate_only and not args.finalize_only:
        verify_equivalence(
            args.equivalence_report,
            args.expected_equivalence_sha256,
            protocol_sha256,
        )
    requests = legacy.read_csv(
        legacy.verify_artifact(protocol["inputs"]["station_day_requests"])
    )
    requests.sort(key=lambda row: (row["date_utc"], int(row["availability_rank"])))
    acquisition_rows = legacy.read_csv(
        legacy.verify_artifact(protocol["inputs"]["acquisition_status"])
    )
    acquisitions = {legacy.station_day_id(row): row for row in acquisition_rows}
    if args.finalize_only:
        verify_equivalence(
            args.equivalence_report,
            args.expected_equivalence_sha256,
            protocol_sha256,
        )
        legacy.finalize(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            requests=requests,
            acquisitions=acquisitions,
        )
        return
    orientations = legacy.orientation_groups(
        legacy.read_csv(
            legacy.verify_artifact(protocol["inputs"]["station_day_orientations"])
        )
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
        device_name="cpu",
        torch_threads=int(protocol["execution"]["torch_threads_per_process"]),
    )
    prepare_hybrid_ensemble(ensemble)
    output_root = Path(protocol["execution"]["output_root"])
    status_path = output_root / "status.jsonl"
    processed = 0
    for index, request in enumerate(requests):
        acquisition = acquisitions[legacy.station_day_id(request)]
        if acquisition["status"] == "download_or_parse_error":
            continue
        if args.limit and processed >= args.limit:
            break
        if args.gate_only and processed >= 1:
            break
        identity = legacy.station_day_id(request)
        source_path = Path(acquisition["mseed_file"])
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(acquisition["mseed_bytes"])
            or sha256_file(source_path) != acquisition["mseed_sha256"]
        ):
            raise RuntimeError(f"Continuous waveform drifted: {source_path}")
        qc_path = Path(acquisition["qc_file"])
        input_identity = legacy.stable_hash(
            {
                "request": request,
                "acquisition": acquisition,
                "qc_sha256": sha256_file(qc_path),
                "orientation_sha256": protocol["inputs"][
                    "station_day_orientations"
                ]["sha256"],
            }
        )
        root = legacy.unit_root(output_root, request)
        summary_path = root / "summary.json"
        if legacy.summary_valid(
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
        memory = mps_memory()
        rss = legacy.maximum_resident_set_bytes()
        limits = protocol["execution"]["memory_guard"]
        if (
            rss > int(limits["maximum_process_rss_bytes"])
            or memory["driver_allocated_bytes"]
            > int(limits["maximum_mps_driver_allocated_bytes"])
        ):
            raise MemoryError(
                f"Hybrid memory guard exceeded: rss={rss}, mps={memory}"
            )
        cache_path = root / "scores.npz"
        legacy.write_npz_atomic(cache_path, arrays)
        summary = {
            "protocol_sha256": protocol_sha256,
            "station_day_id": identity,
            "date_utc": request["date_utc"],
            "station_key": legacy.station_key(request),
            "input_identity": input_identity,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "acquisition_status": acquisition["status"],
            "acquisition_minimum_component_available_fraction": float(
                acquisition["minimum_component_available_fraction"]
            ),
            "duration": duration,
            "metadata": metadata,
            "score_cache": legacy.artifact(cache_path),
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_resident_set_bytes": rss,
            "mps_memory": memory,
        }
        legacy.write_json_atomic(summary_path, summary)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "completed_at_utc": summary["completed_at_utc"],
                        "station_day_id": identity,
                        "summary": legacy.artifact(summary_path),
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
            f"rss={rss / 2**30:.2f} GiB, "
            f"mps={memory['driver_allocated_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )
        processed += 1
        del arrays, metadata
        gc.collect()
        release_hybrid_cache()


if __name__ == "__main__":
    main()
