#!/usr/bin/env python3
"""Download frozen NCSN event windows without model inference."""
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
    TARGET_ROOT / "metadata/manifests/event_station_requests.csv"
)
DEFAULT_OUTPUT_ROOT = TARGET_ROOT / "waveforms/events"
DATASELECT_URL = "https://service.ncedc.org/fdsnws/dataselect/1/query"
USER_AGENT = "CSPP-NCSN-model-blind-event-acquisition/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--limit-events", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
                latest[str(row["event_id"])] = row
    return latest


def append_status(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def request_body(rows: list[dict[str, str]]) -> bytes:
    lines: list[str] = []
    for row in sorted(rows, key=lambda item: int(item["station_rank"])):
        for channel in row["channels"].split(","):
            lines.append(
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
            )
    return ("\n".join(lines) + "\n").encode("ascii")


def fetch_mseed(body: bytes, timeout_seconds: float) -> bytes:
    errors: list[str] = []
    with requests.Session() as session:
        for attempt in range(1, 6):
            try:
                response = session.post(
                    DATASELECT_URL,
                    data=body,
                    headers={
                        "Content-Type": "text/plain",
                        "User-Agent": USER_AGENT,
                    },
                    timeout=(30.0, timeout_seconds),
                )
                response.raise_for_status()
                if not response.content:
                    raise RuntimeError("empty DataSelect response")
                return response.content
            except Exception as exc:  # noqa: BLE001
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
        return {"covered": False, "reason": "missing", "segments": 0}
    if any(
        abs(float(trace.stats.sampling_rate) - sample_rate_hz) > 1e-6
        for trace in traces
    ):
        return {
            "covered": False,
            "reason": "sample_rate_mismatch",
            "segments": len(traces),
        }
    intervals = sorted(
        (
            trace.stats.starttime.datetime.replace(tzinfo=UTC),
            trace.stats.endtime.datetime.replace(tzinfo=UTC)
            + timedelta(seconds=1.0 / sample_rate_hz),
        )
        for trace in traces
    )
    tolerance = timedelta(seconds=max(0.025, 2.5 / sample_rate_hz))
    if intervals[0][0] > required_start + tolerance:
        return {
            "covered": False,
            "reason": "late_start",
            "segments": len(traces),
        }
    merged_end = intervals[0][1]
    gap_count = 0
    maximum_gap = 0.0
    for start, end in intervals[1:]:
        if start > merged_end + tolerance:
            gap_count += 1
            maximum_gap = max(maximum_gap, (start - merged_end).total_seconds())
        merged_end = max(merged_end, end)
    if gap_count:
        return {
            "covered": False,
            "reason": "internal_gap",
            "segments": len(traces),
            "gap_count": gap_count,
            "maximum_gap_seconds": maximum_gap,
        }
    if merged_end < required_end - tolerance:
        return {
            "covered": False,
            "reason": "early_end",
            "segments": len(traces),
        }
    return {
        "covered": True,
        "reason": "complete",
        "segments": len(traces),
        "gap_count": 0,
        "maximum_gap_seconds": 0.0,
    }


def qc_payload(
    payload: bytes,
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stream = read(io.BytesIO(payload), format="MSEED", headonly=True)
    parser_warnings = sorted({str(item.message) for item in caught})
    by_key: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for trace in stream:
        by_key[
            (
                str(trace.stats.network),
                str(trace.stats.station),
                str(trace.stats.location),
                str(trace.stats.channel),
            )
        ].append(trace)

    station_qc: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["station_rank"])):
        start = parse_utc(row["request_start_utc"])
        end = parse_utc(row["request_end_utc"])
        rate = float(row["native_sample_rate_hz"])
        component_qc: dict[str, dict[str, Any]] = {}
        for channel in row["channels"].split(","):
            component_qc[channel] = channel_coverage(
                by_key.get(
                    (
                        row["network"],
                        row["station"],
                        row["location"],
                        channel,
                    ),
                    [],
                ),
                required_start=start,
                required_end=end,
                sample_rate_hz=rate,
            )
        station_qc.append(
            {
                "event_id": row["event_id"],
                "station_rank": int(row["station_rank"]),
                "network": row["network"],
                "station": row["station"],
                "location": row["location"],
                "native_sample_rate_hz": rate,
                "complete": all(item["covered"] for item in component_qc.values()),
                "component_qc": component_qc,
            }
        )
    return station_qc, {
        "trace_segments": len(stream),
        "parser_warnings": parser_warnings,
    }


def process_event(
    event_id: str,
    rows: list[dict[str, str]],
    output_root: Path,
    timeout_seconds: float,
    sleep_seconds: float,
) -> dict[str, Any]:
    body = request_body(rows)
    event_dir = output_root / event_id
    request_path = event_dir / "request.txt"
    mseed_path = event_dir / "waveforms.mseed"
    qc_path = event_dir / "qc.json"
    payload: bytes | None = None
    station_qc: list[dict[str, Any]] = []
    transport_qc: dict[str, Any] = {}
    reason = ""
    try:
        payload = fetch_mseed(body, timeout_seconds)
        station_qc, transport_qc = qc_payload(payload, rows)
        complete = sum(row["complete"] for row in station_qc)
        status = "complete" if complete == len(rows) else "partial"
        if complete == 0:
            status = "no_complete_station"
        if status != "complete":
            reason = f"{complete}_of_{len(rows)}_stations_complete"
    except Exception as exc:  # noqa: BLE001
        status = "download_or_parse_error"
        complete = 0
        reason = f"{type(exc).__name__}: {exc}"

    event_dir.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(body)
    if payload is not None:
        temporary = mseed_path.with_name(f".{mseed_path.name}.tmp-{os.getpid()}")
        temporary.write_bytes(payload)
        temporary.replace(mseed_path)
    write_json_atomic(
        qc_path,
        {
            "event_id": event_id,
            "request": {
                "path": str(request_path),
                "bytes": len(body),
                "sha256": sha256_bytes(body),
            },
            "waveform": (
                {
                    "path": str(mseed_path),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
                if payload is not None
                else None
            ),
            "requested_stations": len(rows),
            "complete_stations": complete,
            "station_qc": station_qc,
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
        "event_id": event_id,
        "origin_time_utc": rows[0]["origin_time_utc"],
        "magnitude": float(rows[0]["magnitude"]),
        "requested_stations": len(rows),
        "complete_stations": complete,
        "status": status,
        "reason": reason,
        "request_sha256": sha256_bytes(body),
        "mseed_file": str(mseed_path) if payload is not None else "",
        "mseed_bytes": len(payload) if payload is not None else 0,
        "mseed_sha256": sha256_bytes(payload) if payload is not None else "",
        "qc_file": str(qc_path),
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

    rows = load_csv(REQUEST_MANIFEST)
    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)
    events = sorted(
        by_event,
        key=lambda event_id: (
            by_event[event_id][0]["origin_time_utc"],
            event_id,
        ),
    )
    if args.limit_events is not None:
        if args.limit_events <= 0:
            raise ValueError("--limit-events must be positive")
        events = events[: args.limit_events]

    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.jsonl"
    latest = load_latest_status(status_path)
    terminal = {"complete", "partial", "no_complete_station"}
    pending: list[tuple[int, str]] = []
    for index, event_id in enumerate(events, start=1):
        previous = latest.get(event_id)
        if previous is not None and previous.get("status") in terminal:
            print(
                f"[{index}/{len(events)}] {event_id}: resume {previous['status']}",
                flush=True,
            )
            continue
        if (
            previous is not None
            and previous.get("status") == "download_or_parse_error"
            and not args.retry_errors
        ):
            print(
                f"[{index}/{len(events)}] {event_id}: retained prior error",
                flush=True,
            )
            continue
        pending.append((index, event_id))

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_event,
                    event_id,
                    by_event[event_id],
                    output_root,
                    args.timeout_seconds,
                    max(0.0, args.sleep_seconds),
                ): (index, event_id)
                for index, event_id in pending
            }
            for future in as_completed(futures):
                index, event_id = futures[future]
                row = future.result()
                append_status(status_path, row)
                latest[event_id] = row
                print(
                    f"[{index}/{len(events)}] {event_id}: {row['status']}, "
                    f"{row['complete_stations']}/{row['requested_stations']} "
                    f"stations, {row['mseed_bytes']} bytes",
                    flush=True,
                )

    latest_rows = [latest[event_id] for event_id in events if event_id in latest]
    fields = [
        "event_id",
        "origin_time_utc",
        "magnitude",
        "requested_stations",
        "complete_stations",
        "status",
        "reason",
        "request_sha256",
        "mseed_file",
        "mseed_bytes",
        "mseed_sha256",
        "qc_file",
        "checked_at_utc",
    ]
    write_csv_atomic(output_root / "latest_status.csv", latest_rows, fields)
    summary = {
        "summary_id": "cspp_v27_ncedc_event_acquisition_v1",
        "created_at_utc": iso_utc(datetime.now(UTC)),
        "event_manifest_freeze_sha256": EXPECTED_EVENT_FREEZE_SHA256,
        "continuous_manifest_freeze_sha256": EXPECTED_CONTINUOUS_FREEZE_SHA256,
        "requested_events": len(events),
        "requested_event_station_records": sum(len(by_event[event]) for event in events),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in latest_rows).items())
        ),
        "complete_event_station_records": sum(
            int(row["complete_stations"]) for row in latest_rows
        ),
        "data_bytes": sum(int(row["mseed_bytes"]) for row in latest_rows),
        "source": DATASELECT_URL,
        "target_waveforms_accessed": bool(latest_rows),
        "model_outputs_computed": False,
    }
    write_json_atomic(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
