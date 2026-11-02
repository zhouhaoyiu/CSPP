#!/usr/bin/env python3
"""Build and seal the NCSN event request manifest before waveform access."""
from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

from obspy import read_events, read_inventory
from obspy.core.event import Event, Origin, Pick
from obspy.core.inventory import Channel, Inventory
from obspy.geodetics import gps2dist_azimuth
import requests


ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path("workspace")
TARGET_ROOT = WORK_ROOT / "ncedc_model_blind_20260817"
RAW_ROOT = TARGET_ROOT / "metadata/raw"
TABLE_ROOT = TARGET_ROOT / "metadata/manifests"
PROTOCOL = (
    ROOT
    / "experiments/cspp_v27_ncedc_model_blind_protocol_20260817.json"
)
EXPECTED_PROTOCOL_SHA256 = (
    "c27b64b7b336e6e96b2f6c61c3a5f8795d7ce3daf0a4ac3c4cd5ccd2df5631ce"
)
LOCAL_FREEZE = (
    ROOT
    / "experiments/cspp_v27_ncedc_event_manifest_freeze_20260817.json"
)
EXTERNAL_FREEZE = TARGET_ROOT / "event_manifest_freeze.json"

EVENT_URL = "https://service.ncedc.org/fdsnws/event/1/query"
STATION_URL = "https://service.ncedc.org/fdsnws/station/1/query"
START = "2024-01-01T00:00:00Z"
END = "2026-01-01T00:00:00Z"
MINIMUM_MAGNITUDE = 4.0
MAXIMUM_DISTANCE_KM = 200.0
MAXIMUM_STATIONS_PER_EVENT = 12
ACCEPTED_SAMPLE_RATES = {100.0, 200.0}
COMPONENTS = ("HNZ", "HNN", "HNE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_no_clobber(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_csv_no_clobber(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def fetch_once(
    session: requests.Session,
    *,
    url: str,
    params: dict[str, Any],
    output: Path,
    request_record: Path,
) -> None:
    if output.exists() or request_record.exists():
        if output.is_file() and request_record.is_file():
            return
        raise RuntimeError(f"Incomplete prior fetch: {output}")

    last_error: BaseException | None = None
    for attempt in range(1, 6):
        try:
            response = session.get(url, params=params, timeout=(30, 600))
            response.raise_for_status()
            if not response.content:
                raise RuntimeError(f"Empty metadata response: {response.url}")
            write_no_clobber(output, response.content)
            write_no_clobber(
                request_record,
                json_bytes(
                    {
                        "requested_at_utc": datetime.now(UTC).isoformat(),
                        "request_url": response.url,
                        "status_code": response.status_code,
                        "response_headers": dict(response.headers),
                        "response": artifact(output),
                    }
                ),
            )
            return
        except BaseException as exc:
            last_error = exc
            if output.exists() or request_record.exists():
                raise
            if attempt < 5:
                time.sleep(2**attempt)
    raise RuntimeError(f"Metadata request failed: {url}") from last_error


def resource_suffix(value: Any) -> str:
    text = str(value or "")
    return text.rstrip("/").rsplit("/", 1)[-1]


def iso(value: Any) -> str:
    return value.datetime.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def preferred_origin(event: Event) -> Origin | None:
    origin = event.preferred_origin()
    return origin if origin is not None else None


def preferred_magnitude(event: Event) -> Any:
    return event.preferred_magnitude()


def is_manual_reviewed_p(pick: Pick, phase: str | None) -> bool:
    phase_text = str(phase or pick.phase_hint or "").upper()
    return (
        phase_text.startswith("P")
        and str(pick.evaluation_mode or "").lower() == "manual"
        and str(pick.evaluation_status or "").lower() == "reviewed"
    )


def pick_lookup(event: Event) -> dict[str, Pick]:
    return {str(pick.resource_id): pick for pick in event.picks}


def active(channel: Channel, epoch: Any) -> bool:
    if channel.start_date is not None and epoch < channel.start_date:
        return False
    if channel.end_date is not None and epoch > channel.end_date:
        return False
    return True


def channel_groups(inventory: Inventory, epoch: Any) -> dict[str, list[dict[str, Any]]]:
    by_station: dict[str, dict[str, dict[str, Channel]]] = {}
    for network in inventory:
        if network.code != "NC":
            continue
        for station in network:
            for channel in station:
                if channel.code not in COMPONENTS or not active(channel, epoch):
                    continue
                location = channel.location_code or ""
                by_station.setdefault(station.code, {}).setdefault(location, {})[
                    channel.code
                ] = channel

    result: dict[str, list[dict[str, Any]]] = {}
    for station, locations in by_station.items():
        for location, channels in locations.items():
            if set(channels) != set(COMPONENTS):
                continue
            rates = [float(channels[code].sample_rate) for code in COMPONENTS]
            if max(rates) - min(rates) > 1e-6:
                continue
            rate = rates[0]
            if rate not in ACCEPTED_SAMPLE_RATES:
                continue
            latitudes = [float(channels[code].latitude) for code in COMPONENTS]
            longitudes = [float(channels[code].longitude) for code in COMPONENTS]
            result.setdefault(station, []).append(
                {
                    "location": location,
                    "channels": ",".join(COMPONENTS),
                    "sample_rate_hz": rate,
                    "latitude": sum(latitudes) / len(latitudes),
                    "longitude": sum(longitudes) / len(longitudes),
                }
            )
    for groups in result.values():
        groups.sort(key=lambda item: (item["location"], item["sample_rate_hz"]))
    return result


def choose_arrival(
    rows: list[dict[str, Any]],
    *,
    require_positive_weight: bool,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if not require_positive_weight or float(row["arrival_weight"]) > 0.0
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            -float(row["arrival_weight"]),
            abs(float(row["time_residual_seconds"])),
            row["pick_time_utc"],
            row["pick_id"],
        ),
    )


def event_rows(
    catalog: Any,
    inventory: Inventory,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events_out: list[dict[str, Any]] = []
    stations_out: list[dict[str, Any]] = []

    events = sorted(
        catalog,
        key=lambda event: (
            event.preferred_origin().time
            if event.preferred_origin() is not None
            else float("inf")
        ),
    )
    for event_index, event in enumerate(events):
        event_id = resource_suffix(event.resource_id)
        origin = preferred_origin(event)
        magnitude = preferred_magnitude(event)
        event_type = str(event.event_type or "").lower()
        base = {
            "event_index": event_index,
            "event_id": event_id,
            "origin_time_utc": "",
            "latitude": "",
            "longitude": "",
            "depth_km": "",
            "magnitude": "",
            "magnitude_type": "",
            "event_type": event_type,
            "preferred_origin_arrivals": 0,
            "manual_reviewed_p_arrivals": 0,
            "positive_weight_manual_reviewed_p_arrivals": 0,
            "eligible_station_count": 0,
            "selected_station_count": 0,
            "status": "excluded",
            "exclusion_reason": "",
        }
        if origin is None or magnitude is None:
            base["exclusion_reason"] = "missing_preferred_origin_or_magnitude"
            events_out.append(base)
            continue
        base.update(
            {
                "origin_time_utc": iso(origin.time),
                "latitude": float(origin.latitude),
                "longitude": float(origin.longitude),
                "depth_km": float(origin.depth or 0.0) / 1000.0,
                "magnitude": float(magnitude.mag),
                "magnitude_type": str(magnitude.magnitude_type or ""),
                "preferred_origin_arrivals": len(origin.arrivals),
            }
        )
        if event_type != "earthquake":
            base["exclusion_reason"] = "event_type_not_earthquake"
            events_out.append(base)
            continue
        if float(magnitude.mag) < MINIMUM_MAGNITUDE:
            base["exclusion_reason"] = "magnitude_below_threshold"
            events_out.append(base)
            continue

        picks = pick_lookup(event)
        arrivals_by_station: dict[str, list[dict[str, Any]]] = {}
        for arrival in origin.arrivals:
            pick = picks.get(str(arrival.pick_id))
            if pick is None or not is_manual_reviewed_p(pick, arrival.phase):
                continue
            waveform_id = pick.waveform_id
            if waveform_id is None or waveform_id.network_code != "NC":
                continue
            station = waveform_id.station_code or ""
            if not station:
                continue
            weight = float(arrival.time_weight or 0.0)
            residual = float(arrival.time_residual or 0.0)
            arrivals_by_station.setdefault(station, []).append(
                {
                    "station": station,
                    "pick_id": resource_suffix(pick.resource_id),
                    "pick_time_utc": iso(pick.time),
                    "pick_channel": waveform_id.channel_code or "",
                    "pick_location": waveform_id.location_code or "",
                    "arrival_weight": weight,
                    "time_residual_seconds": residual,
                }
            )

        all_reviewed = sum(len(rows) for rows in arrivals_by_station.values())
        positive_reviewed = sum(
            sum(float(row["arrival_weight"]) > 0.0 for row in rows)
            for rows in arrivals_by_station.values()
        )
        base["manual_reviewed_p_arrivals"] = all_reviewed
        base["positive_weight_manual_reviewed_p_arrivals"] = positive_reviewed

        groups = channel_groups(inventory, origin.time)
        eligible: list[dict[str, Any]] = []
        for station, arrivals in arrivals_by_station.items():
            reference = choose_arrival(arrivals, require_positive_weight=True)
            secondary = choose_arrival(arrivals, require_positive_weight=False)
            if reference is None or secondary is None or station not in groups:
                continue
            channel_group = groups[station][0]
            distance_m, _, _ = gps2dist_azimuth(
                float(origin.latitude),
                float(origin.longitude),
                channel_group["latitude"],
                channel_group["longitude"],
            )
            distance_km = distance_m / 1000.0
            if distance_km > MAXIMUM_DISTANCE_KM:
                continue
            request_start = origin.time - 60.0
            request_end = origin.time + 180.0
            eligible.append(
                {
                    "event_id": event_id,
                    "origin_time_utc": iso(origin.time),
                    "event_latitude": float(origin.latitude),
                    "event_longitude": float(origin.longitude),
                    "depth_km": float(origin.depth or 0.0) / 1000.0,
                    "magnitude": float(magnitude.mag),
                    "magnitude_type": str(magnitude.magnitude_type or ""),
                    "network": "NC",
                    "station": station,
                    "location": channel_group["location"],
                    "channels": channel_group["channels"],
                    "native_sample_rate_hz": channel_group["sample_rate_hz"],
                    "station_latitude": channel_group["latitude"],
                    "station_longitude": channel_group["longitude"],
                    "epicentral_distance_km": distance_km,
                    "primary_pick_id": reference["pick_id"],
                    "primary_pick_time_utc": reference["pick_time_utc"],
                    "primary_arrival_weight": reference["arrival_weight"],
                    "primary_time_residual_seconds": reference[
                        "time_residual_seconds"
                    ],
                    "secondary_pick_id": secondary["pick_id"],
                    "secondary_pick_time_utc": secondary["pick_time_utc"],
                    "request_start_utc": iso(request_start),
                    "request_end_utc": iso(request_end),
                }
            )

        eligible.sort(
            key=lambda row: (
                row["epicentral_distance_km"],
                row["network"],
                row["station"],
                row["location"],
                row["channels"],
            )
        )
        base["eligible_station_count"] = len(eligible)
        if len(eligible) < 3:
            base["exclusion_reason"] = "fewer_than_three_primary_eligible_stations"
            events_out.append(base)
            continue
        selected = eligible[:MAXIMUM_STATIONS_PER_EVENT]
        for station_rank, row in enumerate(selected, start=1):
            row["station_rank"] = station_rank
            stations_out.append(row)
        base["selected_station_count"] = len(selected)
        base["status"] = "selected"
        events_out.append(base)
    return events_out, stations_out


def main() -> None:
    if LOCAL_FREEZE.exists() or EXTERNAL_FREEZE.exists():
        raise FileExistsError("NCSN event request manifest is already sealed")
    if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Frozen NCSN evaluation protocol drifted")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != (
        "frozen_before_target_period_metadata_or_waveform_access"
    ):
        raise RuntimeError("Unexpected NCSN protocol status")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "CSPP-NCSN-model-blind-evaluation/1.0"}
    )
    event_xml = RAW_ROOT / "ncedc_events_2024_2025_m4_arrivals.xml"
    event_request = RAW_ROOT / "ncedc_events_2024_2025_m4_arrivals.request.json"
    station_xml = RAW_ROOT / "ncsn_hn_response_2024_2025.xml"
    station_request = RAW_ROOT / "ncsn_hn_response_2024_2025.request.json"
    fetch_once(
        session,
        url=EVENT_URL,
        params={
            "starttime": START,
            "endtime": END,
            "minmagnitude": MINIMUM_MAGNITUDE,
            "includearrivals": "true",
            "orderby": "time-asc",
            "format": "xml",
        },
        output=event_xml,
        request_record=event_request,
    )
    fetch_once(
        session,
        url=STATION_URL,
        params={
            "network": "NC",
            "channel": "HN?",
            "starttime": START,
            "endtime": END,
            "level": "response",
            "format": "xml",
        },
        output=station_xml,
        request_record=station_request,
    )

    catalog = read_events(BytesIO(event_xml.read_bytes()), format="QUAKEML")
    inventory = read_inventory(BytesIO(station_xml.read_bytes()), format="STATIONXML")
    events, stations = event_rows(catalog, inventory)
    event_table = TABLE_ROOT / "event_catalog_and_selection.csv"
    station_table = TABLE_ROOT / "event_station_requests.csv"
    event_fields = [
        "event_index",
        "event_id",
        "origin_time_utc",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "magnitude_type",
        "event_type",
        "preferred_origin_arrivals",
        "manual_reviewed_p_arrivals",
        "positive_weight_manual_reviewed_p_arrivals",
        "eligible_station_count",
        "selected_station_count",
        "status",
        "exclusion_reason",
    ]
    station_fields = [
        "event_id",
        "origin_time_utc",
        "event_latitude",
        "event_longitude",
        "depth_km",
        "magnitude",
        "magnitude_type",
        "network",
        "station",
        "location",
        "channels",
        "native_sample_rate_hz",
        "station_latitude",
        "station_longitude",
        "epicentral_distance_km",
        "primary_pick_id",
        "primary_pick_time_utc",
        "primary_arrival_weight",
        "primary_time_residual_seconds",
        "secondary_pick_id",
        "secondary_pick_time_utc",
        "request_start_utc",
        "request_end_utc",
        "station_rank",
    ]
    write_csv_no_clobber(event_table, event_fields, events)
    write_csv_no_clobber(station_table, station_fields, stations)

    selected_events = [row for row in events if row["status"] == "selected"]
    exclusion_counts: dict[str, int] = {}
    for row in events:
        reason = str(row["exclusion_reason"] or "selected")
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    payload = {
        "freeze_id": "cspp_v27_ncedc_event_manifest_freeze_v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "status": "event_station_requests_frozen_before_waveform_access",
        "protocol": artifact(PROTOCOL),
        "source_metadata": {
            "events": artifact(event_xml),
            "event_request": artifact(event_request),
            "stations": artifact(station_xml),
            "station_request": artifact(station_request),
        },
        "manifests": {
            "event_catalog_and_selection": artifact(event_table),
            "event_station_requests": artifact(station_table),
        },
        "counts": {
            "catalog_events": len(events),
            "selected_events": len(selected_events),
            "selected_event_station_requests": len(stations),
            "unique_selected_stations": len(
                {row["station"] for row in stations}
            ),
            "native_sample_rate_hz": {
                str(rate): sum(
                    float(row["native_sample_rate_hz"]) == rate
                    for row in stations
                )
                for rate in sorted(ACCEPTED_SAMPLE_RATES)
            },
            "event_selection_outcomes": exclusion_counts,
        },
        "deterministic_tie_breaks": {
            "arrival": (
                "descending arrival weight, ascending absolute residual, pick "
                "time, and pick identifier"
            ),
            "channel_group": "ascending location code and sample rate",
            "station": (
                "ascending epicentral distance, network, station, location, "
                "and channel list"
            ),
        },
        "evidence_boundary": {
            "target_metadata_accessed_after_protocol_freeze": True,
            "target_waveforms_accessed": False,
            "model_outputs_computed": False,
            "selection_uses_waveform_amplitude_or_model_output": False,
            "unavailable_requests_may_be_replaced": False,
        },
        "code": artifact(Path(__file__).resolve()),
        "next_boundary": (
            "Waveform acquisition may begin only after the downloader verifies "
            "this freeze. It must attempt every request exactly once per retry "
            "policy and preserve missing, partial, and rejected outcomes."
        ),
    }
    content = json_bytes(payload)
    published: list[Path] = []
    try:
        for destination in (LOCAL_FREEZE, EXTERNAL_FREEZE):
            write_no_clobber(destination, content)
            published.append(destination)
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    if sha256_file(LOCAL_FREEZE) != sha256_file(EXTERNAL_FREEZE):
        raise RuntimeError("Local and external event manifest freezes differ")
    print(
        json.dumps(
            {
                "local_freeze": str(LOCAL_FREEZE),
                "external_freeze": str(EXTERNAL_FREEZE),
                "sha256": sha256_file(LOCAL_FREEZE),
                "counts": payload["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
