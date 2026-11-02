#!/usr/bin/env python3
"""Apply frozen NCSN catalog timing checks to continuous CSPP candidates."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

from obspy import read_events
from obspy.geodetics.base import locations2degrees
from obspy.taup import TauPyModel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cspp_v27_reused_core_score_cache import sha256_file  # noqa: E402


PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_continuous_catalog_adjudication_protocol_20260817.json"
)


def verify(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != spec["sha256"]
    ):
        raise RuntimeError(f"Artifact drifted: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
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


def event_id(event: Any) -> str:
    return str(event.resource_id).rsplit("/", 1)[-1]


def load_events(summary: dict[str, Any]) -> list[dict[str, Any]]:
    events_by_id: dict[str, dict[str, Any]] = {}
    for snapshot in summary["snapshots"]:
        catalog = read_events(str(verify(snapshot["response"])))
        if len(catalog) != int(snapshot["event_count"]):
            raise RuntimeError(f"Catalog event count differs: {snapshot['date_utc']}")
        for event in catalog:
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude()
            if magnitude is None and event.magnitudes:
                magnitude = event.magnitudes[0]
            row = {
                "event_id": event_id(event),
                "origin_epoch": float(origin.time.timestamp),
                "origin_utc": str(origin.time),
                "latitude": float(origin.latitude),
                "longitude": float(origin.longitude),
                "depth_km": max(float(origin.depth or 0.0) / 1000.0, 0.0),
                "magnitude": (
                    None if magnitude is None else float(magnitude.mag)
                ),
                "event_type": str(event.event_type or "unknown"),
            }
            previous = events_by_id.get(row["event_id"])
            if previous is not None and previous != row:
                raise RuntimeError(f"Catalog event differs: {row['event_id']}")
            events_by_id[row["event_id"]] = row
    return sorted(events_by_id.values(), key=lambda row: row["origin_epoch"])


def first_p_seconds(
    taup: TauPyModel,
    event: dict[str, Any],
    latitude: float,
    longitude: float,
) -> float | None:
    distance = locations2degrees(
        event["latitude"], event["longitude"], latitude, longitude
    )
    arrivals = taup.get_travel_times(
        source_depth_in_km=min(float(event["depth_km"]), 700.0),
        distance_in_degree=float(distance),
        phase_list=["P", "p", "Pn", "Pg"],
    )
    return min((float(arrival.time) for arrival in arrivals), default=None)


def match_candidate(
    row: dict[str, str],
    *,
    events: list[dict[str, Any]],
    event_epochs: list[float],
    taup: TauPyModel,
    lookback: float,
    tolerance: float,
    cache: dict[tuple[str, float, float], float | None],
) -> dict[str, Any] | None:
    estimated = float(row["estimated_p_epoch"])
    latitude = float(row["station_latitude"])
    longitude = float(row["station_longitude"])
    first = bisect_left(event_epochs, estimated - lookback)
    last = bisect_left(event_epochs, estimated + tolerance)
    selected = None
    for event in events[first:last]:
        key = (event["event_id"], round(latitude, 6), round(longitude, 6))
        if key not in cache:
            cache[key] = first_p_seconds(taup, event, latitude, longitude)
        travel = cache[key]
        if travel is None:
            continue
        predicted = float(event["origin_epoch"]) + float(travel)
        residual = estimated - predicted
        if abs(residual) > tolerance:
            continue
        candidate = {
            "event": event,
            "predicted_p_epoch": predicted,
            "residual_seconds": residual,
        }
        if selected is None or abs(residual) < abs(selected["residual_seconds"]):
            selected = candidate
    return selected


def main() -> None:
    expected = os.environ.get("CSPP_NCEDC_CATALOG_ADJUDICATION_SHA256", "")
    actual = sha256_file(PROTOCOL)
    if not expected or actual != expected:
        raise RuntimeError(f"Set the frozen adjudication SHA-256; actual={actual}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for relative, digest in protocol["code_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Catalog adjudication code drifted: {relative}")
    catalog_summary_path = Path(protocol["inputs"]["catalog_completion_path"])
    catalog_summary = json.loads(catalog_summary_path.read_text(encoding="utf-8"))
    if catalog_summary.get("protocol_sha256") != protocol["inputs"][
        "catalog_transport_sha256"
    ]:
        raise RuntimeError("Catalog completion protocol differs")
    events = load_events(catalog_summary)
    event_epochs = [float(event["origin_epoch"]) for event in events]
    analysis_result_path = Path(protocol["inputs"]["continuous_result_path"])
    analysis_result = json.loads(analysis_result_path.read_text(encoding="utf-8"))
    if analysis_result.get("protocol_sha256") != protocol["inputs"][
        "continuous_analysis_sha256"
    ]:
        raise RuntimeError("Continuous analysis protocol differs")
    candidate_path = verify(analysis_result["artifacts"]["primary_candidates.csv"])
    group_path = verify(analysis_result["artifacts"]["network_groups.csv"])
    candidates = read_csv(candidate_path)
    groups = read_csv(group_path)
    matching = protocol["matching"]
    taup = TauPyModel(model=matching["travel_time_model"])
    cache: dict[tuple[str, float, float], float | None] = {}
    matches: dict[str, str] = {}
    output_rows = []
    counts = Counter()
    fields = list(candidates[0]) if candidates else []
    added = [
        "catalog_class",
        "catalog_event_id",
        "catalog_event_type",
        "catalog_magnitude",
        "catalog_origin_utc",
        "predicted_p_epoch",
        "catalog_residual_seconds",
        "catalog_interpretation",
    ]
    for row in candidates:
        match = match_candidate(
            row,
            events=events,
            event_epochs=event_epochs,
            taup=taup,
            lookback=float(matching["origin_lookback_seconds"]),
            tolerance=float(matching["taup_tolerance_seconds"]),
            cache=cache,
        )
        if match is None:
            catalog = {
                "catalog_class": "unresolved",
                "catalog_event_id": "",
                "catalog_event_type": "",
                "catalog_magnitude": "",
                "catalog_origin_utc": "",
                "predicted_p_epoch": "",
                "catalog_residual_seconds": "",
                "catalog_interpretation": (
                    "no match under the frozen catalog timing rule; not a false label"
                ),
            }
        else:
            event = match["event"]
            matches[row["candidate_id"]] = event["event_id"]
            catalog = {
                "catalog_class": "catalog_taup_compatible",
                "catalog_event_id": event["event_id"],
                "catalog_event_type": event["event_type"],
                "catalog_magnitude": (
                    "" if event["magnitude"] is None else event["magnitude"]
                ),
                "catalog_origin_utc": event["origin_utc"],
                "predicted_p_epoch": match["predicted_p_epoch"],
                "catalog_residual_seconds": match["residual_seconds"],
                "catalog_interpretation": (
                    "catalog and iasp91 timing compatible; manual review pending"
                ),
            }
        counts[catalog["catalog_class"]] += 1
        output_rows.append({**row, **catalog})

    group_rows = []
    group_fields = list(groups[0]) if groups else []
    group_added = [
        "shared_catalog_event_id",
        "shared_catalog_station_count",
        "catalog_group_class",
    ]
    for group in groups:
        ids = str(group["member_candidate_ids"]).split(";")
        shared = Counter(matches.get(candidate_id, "") for candidate_id in ids)
        shared.pop("", None)
        event, stations = shared.most_common(1)[0] if shared else ("", 0)
        supported = stations >= int(matching["minimum_shared_event_stations"])
        group_rows.append(
            {
                **group,
                "shared_catalog_event_id": event if supported else "",
                "shared_catalog_station_count": stations if supported else 0,
                "catalog_group_class": (
                    "catalog_supported_time_coincidence"
                    if supported
                    else "unresolved_time_coincidence"
                ),
            }
        )
    output_root = Path(protocol["execution"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    write_csv(output_root / "candidates_catalog_screened.csv", output_rows, fields + added)
    write_csv(output_root / "network_groups_catalog_screened.csv", group_rows, group_fields + group_added)
    result = {
        "result_id": "cspp_v27_ncedc_continuous_catalog_adjudication_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": actual,
        "catalog_events": len(events),
        "candidate_count": len(candidates),
        "catalog_class_counts": dict(sorted(counts.items())),
        "network_group_count": len(groups),
        "catalog_supported_network_group_count": sum(
            row["catalog_group_class"] == "catalog_supported_time_coincidence"
            for row in group_rows
        ),
        "manual_review_status": "pending",
        "claim_boundary": (
            "Catalog timing compatibility is provisional. Unmatched candidates "
            "remain unresolved and are not false-alarm labels."
        ),
    }
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
