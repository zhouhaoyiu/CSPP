#!/usr/bin/env python3
"""Download frozen NCSN continuous station-days without model inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any
import warnings

from obspy import read
import requests


ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path("workspace")
TARGET_ROOT = WORK_ROOT / "ncedc_model_blind_20260817"
EVENT_FREEZE = (
    ROOT
    / "experiments/cspp_v27_ncedc_event_manifest_freeze_20260817.json"
)
CONTINUOUS_FREEZE = (
    ROOT
    / "experiments/cspp_v27_ncedc_continuous_manifest_freeze_20260817.json"
)
EXPECTED_EVENT_FREEZE_SHA256 = (
    "e088d5b42c07c0f38be915c8c893b73cedf1bb393d3c1580f0fca78cbf868743"
)
EXPECTED_CONTINUOUS_FREEZE_SHA256 = (
    "1839769549d78d0f756b2b2016a425cbe89bc6216e3a45bed1d0b16b3fcb47f6"
)
REQUEST_MANIFEST = (
    TARGET_ROOT / "metadata/manifests/continuous_station_day_requests.csv"
)
DEFAULT_OUTPUT_ROOT = TARGET_ROOT / "waveforms/continuous"
DATASELECT_URL = "https://service.ncedc.org/fdsnws/dataselect/1/query"
USER_AGENT = "CSPP-NCSN-model-blind-continuous-acquisition/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--limit-station-days", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_latest_status(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row["station_day_id"])] = row
    return latest


def append_status(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def station_day_id(row: dict[str, str]) -> str:
    location = row["location"] or "--"
    return f"{row['date_utc']}__{row['network']}.{row['station']}.{location}"


def request_body(row: dict[str, str]) -> bytes:
    lines = [
        " ".join(
            (
                row["network"],
                row["station"],
                row["location"] or "--",
                channel,
                row["request_start_utc"],
                row["request_end_utc"],
            )
        )
        for channel in row["channels"].split(",")
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def fetch_mseed_to_file(
    body: bytes,
    destination: Path,
    timeout_seconds: float,
) -> tuple[int, str]:
    errors: list[str] = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{attempt}"
        )
        temporary.unlink(missing_ok=True)
        try:
            with requests.Session() as session:
                response = session.post(
                    DATASELECT_URL,
                    data=body,
                    headers={
                        "Content-Type": "text/plain",
                        "User-Agent": USER_AGENT,
                    },
                    timeout=(30.0, timeout_seconds),
                    stream=True,
                )
                response.raise_for_status()
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if size <= 0:
                    raise RuntimeError("empty DataSelect response")
                temporary.replace(destination)
                return size, digest.hexdigest()
        except Exception as exc:  # noqa: BLE001
            temporary.unlink(missing_ok=True)
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < 5:
                time.sleep(min(30.0, float(2**attempt)))
    raise RuntimeError("; ".join(errors))


def channel_coverage(
    traces: list[Any],
    *,
    required_start: datetime,
    required_end: datetime,
    sample_rate_hz: float,
) -> dict[str, Any]:
    if not traces:
        return {
            "covered": False,
            "reason": "missing",
            "segments": 0,
            "available_seconds": 0.0,
        }
    if any(
        abs(float(trace.stats.sampling_rate) - sample_rate_hz) > 1e-6
        for trace in traces
    ):
        return {
            "covered": False,
            "reason": "sample_rate_mismatch",
            "segments": len(traces),
            "available_seconds": 0.0,
        }
    intervals = sorted(
        (
            max(
                trace.stats.starttime.datetime.replace(tzinfo=UTC),
                required_start,
            ),
            min(
                trace.stats.endtime.datetime.replace(tzinfo=UTC)
                + timedelta(seconds=1.0 / sample_rate_hz),
                required_end,
            ),
        )
        for trace in traces
    )
    intervals = [(start, end) for start, end in intervals if end > start]
    if not intervals:
        return {
            "covered": False,
            "reason": "outside_request",
            "segments": len(traces),
            "available_seconds": 0.0,
        }
    tolerance = timedelta(seconds=max(0.025, 2.5 / sample_rate_hz))
    merged: list[list[datetime]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    available_seconds = sum((end - start).total_seconds() for start, end in merged)
    complete = (
        merged[0][0] <= required_start + tolerance
        and merged[-1][1] >= required_end - tolerance
        and len(merged) == 1
    )
    return {
        "covered": complete,
        "reason": "complete" if complete else "partial_or_gappy",
        "segments": len(traces),
        "merged_intervals": len(merged),
        "available_seconds": available_seconds,
        "available_fraction": available_seconds
        / (required_end - required_start).total_seconds(),
    }


def qc_file(path: Path, row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stream = read(str(path), format="MSEED", headonly=True)
    by_channel: dict[str, list[Any]] = defaultdict(list)
    for trace in stream:
        if (
            str(trace.stats.network) == row["network"]
            and str(trace.stats.station) == row["station"]
            and str(trace.stats.location) == row["location"]
        ):
            by_channel[str(trace.stats.channel)].append(trace)
    start = parse_utc(row["request_start_utc"])
    end = parse_utc(row["request_end_utc"])
    rate = float(row["native_sample_rate_hz"])
    component_qc = {
        channel: channel_coverage(
            by_channel.get(channel, []),
            required_start=start,
            required_end=end,
            sample_rate_hz=rate,
        )
        for channel in row["channels"].split(",")
    }
    minimum_fraction = min(
        float(item.get("available_fraction", 0.0)) for item in component_qc.values()
    )
    complete = all(item["covered"] for item in component_qc.values())
    return component_qc, {
        "complete": complete,
        "minimum_component_available_fraction": minimum_fraction,
        "trace_segments": len(stream),
        "parser_warnings": sorted({str(item.message) for item in caught}),
    }


def process_station_day(
    row: dict[str, str],
    output_root: Path,
    timeout_seconds: float,
    sleep_seconds: float,
) -> dict[str, Any]:
    identity = station_day_id(row)
    location = row["location"] or "--"
    station_dir = output_root / row["date_utc"] / f"{row['network']}.{row['station']}.{location}"
    request_path = station_dir / "request.txt"
    mseed_path = station_dir / "waveforms.mseed"
    qc_path = station_dir / "qc.json"
    body = request_body(row)
    station_dir.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(body)
    try:
        size, digest = fetch_mseed_to_file(body, mseed_path, timeout_seconds)
        component_qc, transport_qc = qc_file(mseed_path, row)
        status = "complete" if transport_qc["complete"] else "partial"
        reason = "" if status == "complete" else "partial_or_gappy_components"
    except Exception as exc:  # noqa: BLE001
        mseed_path.unlink(missing_ok=True)
        size = 0
        digest = ""
        component_qc = {}
        transport_qc = {
            "complete": False,
            "minimum_component_available_fraction": 0.0,
            "trace_segments": 0,
            "parser_warnings": [],
        }
        status = "download_or_parse_error"
        reason = f"{type(exc).__name__}: {exc}"
    write_json_atomic(
        qc_path,
        {
            "station_day_id": identity,
            "request_row": row,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "waveform": (
                {"path": str(mseed_path), "bytes": size, "sha256": digest}
                if size
                else None
            ),
            "component_qc": component_qc,
            **transport_qc,
            "status": status,
            "reason": reason,
            "model_outputs_computed": False,
            "checked_at_utc": iso_utc(datetime.now(UTC)),
        },
    )
    if sleep_seconds > 0.0:
        time.sleep(sleep_seconds)
    return {
        "station_day_id": identity,
        "date_utc": row["date_utc"],
        "network": row["network"],
        "station": row["station"],
        "location": row["location"],
        "native_sample_rate_hz": float(row["native_sample_rate_hz"]),
        "status": status,
        "reason": reason,
        "minimum_component_available_fraction": transport_qc[
            "minimum_component_available_fraction"
        ],
        "mseed_file": str(mseed_path) if size else "",
        "mseed_bytes": size,
        "mseed_sha256": digest,
        "qc_file": str(qc_path),
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "checked_at_utc": iso_utc(datetime.now(UTC)),
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")
    if sha256_file(EVENT_FREEZE) != EXPECTED_EVENT_FREEZE_SHA256:
        raise RuntimeError("Frozen NCSN event manifest drifted")
    if sha256_file(CONTINUOUS_FREEZE) != EXPECTED_CONTINUOUS_FREEZE_SHA256:
        raise RuntimeError("Frozen NCSN continuous manifest drifted")
    requests_out = load_csv(REQUEST_MANIFEST)
    requests_out.sort(
        key=lambda row: (row["date_utc"], int(row["availability_rank"]))
    )
    if args.limit_station_days is not None:
        if args.limit_station_days <= 0:
            raise ValueError("--limit-station-days must be positive")
        requests_out = requests_out[: args.limit_station_days]

    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.jsonl"
    latest = load_latest_status(status_path)
    pending: list[tuple[int, dict[str, str]]] = []
    terminal = {"complete", "partial"}
    for index, row in enumerate(requests_out, start=1):
        identity = station_day_id(row)
        previous = latest.get(identity)
        if previous is not None and previous.get("status") in terminal:
            print(
                f"[{index}/{len(requests_out)}] {identity}: resume "
                f"{previous['status']}",
                flush=True,
            )
            continue
        if (
            previous is not None
            and previous.get("status") == "download_or_parse_error"
            and not args.retry_errors
        ):
            print(
                f"[{index}/{len(requests_out)}] {identity}: retained prior error",
                flush=True,
            )
            continue
        pending.append((index, row))

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_station_day,
                    row,
                    output_root,
                    args.timeout_seconds,
                    max(0.0, args.sleep_seconds),
                ): (index, row)
                for index, row in pending
            }
            for future in as_completed(futures):
                index, row = futures[future]
                result = future.result()
                append_status(status_path, result)
                latest[result["station_day_id"]] = result
                print(
                    f"[{index}/{len(requests_out)}] {result['station_day_id']}: "
                    f"{result['status']}, "
                    f"{result['minimum_component_available_fraction']:.4f}, "
                    f"{result['mseed_bytes']} bytes",
                    flush=True,
                )

    latest_rows = [
        latest[station_day_id(row)]
        for row in requests_out
        if station_day_id(row) in latest
    ]
    fields = [
        "station_day_id",
        "date_utc",
        "network",
        "station",
        "location",
        "native_sample_rate_hz",
        "status",
        "reason",
        "minimum_component_available_fraction",
        "mseed_file",
        "mseed_bytes",
        "mseed_sha256",
        "qc_file",
        "request_sha256",
        "checked_at_utc",
    ]
    write_csv_atomic(output_root / "latest_status.csv", latest_rows, fields)
    summary = {
        "summary_id": "cspp_v27_ncedc_continuous_acquisition_v1",
        "created_at_utc": iso_utc(datetime.now(UTC)),
        "event_manifest_freeze_sha256": EXPECTED_EVENT_FREEZE_SHA256,
        "continuous_manifest_freeze_sha256": EXPECTED_CONTINUOUS_FREEZE_SHA256,
        "requested_station_days": len(requests_out),
        "completed_station_days": len(latest_rows),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in latest_rows).items())
        ),
        "complete_station_days": sum(row["status"] == "complete" for row in latest_rows),
        "partial_station_days": sum(row["status"] == "partial" for row in latest_rows),
        "data_bytes": sum(int(row["mseed_bytes"]) for row in latest_rows),
        "source": DATASELECT_URL,
        "target_waveforms_accessed": bool(latest_rows),
        "model_outputs_computed": False,
    }
    write_json_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
