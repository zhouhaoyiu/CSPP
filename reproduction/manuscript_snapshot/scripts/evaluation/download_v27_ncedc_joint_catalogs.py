#!/usr/bin/env python3
"""Download regional and global catalogs for the joint continuous cohort."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

from obspy import read_events
import requests


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import download_v27_ncedc_continuous_waveforms as base  # noqa: E402


DEFAULT_ROOT = Path(
    "workspace/"
    "ncedc_joint_continuous_m3_20260904"
)
REGIONAL_URL = "https://service.ncedc.org/fdsnws/event/1/query"
GLOBAL_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


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


def fetch(session: requests.Session, url: str, params: dict[str, str]) -> requests.Response:
    for attempt in range(1, 6):
        try:
            response = session.get(url, params=params, timeout=(30, 180))
            response.raise_for_status()
            read_events(BytesIO(response.content), format="QUAKEML")
            return response
        except BaseException:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    protocol = json.loads((args.root / "joint_continuous_protocol.json").read_text())
    dates = base.load_csv(Path(protocol["artifacts"]["selected_dates"]["path"]))
    output = args.root / "catalogs"
    session = requests.Session()
    session.headers["User-Agent"] = "CSPP-NCSN-joint-continuous-catalog/1.0"
    snapshots = []
    for index, row in enumerate(dates, start=1):
        day = datetime.fromisoformat(row["date_utc"]).replace(tzinfo=UTC)
        start = day - timedelta(seconds=1800)
        end = day + timedelta(days=1)
        for name, url, extra in (
            ("regional", REGIONAL_URL, {"format": "xml"}),
            ("global_m45", GLOBAL_URL, {"format": "quakeml", "minmagnitude": "4.5"}),
        ):
            directory = output / name
            path = directory / f"{row['date_utc']}.xml"
            request_path = directory / f"{row['date_utc']}.request.json"
            if path.exists() and request_path.exists():
                request = json.loads(request_path.read_text())
                if request["response"]["sha256"] != base.sha256_file(path):
                    raise RuntimeError(f"Catalog snapshot drifted: {path}")
                snapshots.append(request)
                continue
            params = {
                "starttime": base.iso_utc(start),
                "endtime": base.iso_utc(end),
                "orderby": "time-asc",
                **extra,
            }
            response = fetch(session, url, params)
            write_once(path, response.content)
            request = {
                "catalog": name,
                "date_utc": row["date_utc"],
                "requested_at_utc": datetime.now(UTC).isoformat(),
                "request_url": response.url,
                "response": {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": base.sha256_file(path),
                },
                "model_outputs_computed": False,
            }
            write_once(
                request_path,
                (json.dumps(request, indent=2, sort_keys=True) + "\n").encode(),
            )
            snapshots.append(request)
        print(f"[{index}/{len(dates)}] {row['date_utc']}", flush=True)
    summary = {
        "summary_id": "cspp_v27_ncedc_joint_continuous_catalogs_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort_protocol_sha256": base.sha256_file(
            args.root / "joint_continuous_protocol.json"
        ),
        "date_count": len(dates),
        "snapshot_count": len(snapshots),
        "regional_query": "all events",
        "global_query": "M>=4.5",
        "model_outputs_computed": False,
    }
    base.write_json_atomic(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
