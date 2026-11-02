"""Deterministic replay of frozen physical-station candidate rows."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from cspp.association.connected_triplet import ConnectedTripletTravelTimeConfirmer


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Not a boolean value: {value!r}")


def replay_candidate_rows(
    candidates: Iterable[Mapping[str, Any]],
    station_coordinates: Mapping[str, Mapping[str, float]],
    edges: Iterable[Iterable[str]],
    network_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay candidate availability and return confirmations/transitions."""
    confirmer = ConnectedTripletTravelTimeConfirmer(
        station_coordinates,
        edges,
        required_stations=int(network_contract["required_stations"]),
        association_window_seconds=float(
            network_contract["association_window_seconds"]
        ),
        minimum_p_velocity_km_s=float(
            network_contract["minimum_apparent_velocity_km_s"]
        ),
        travel_time_tolerance_seconds=float(
            network_contract["travel_time_tolerance_seconds"]
        ),
        station_refractory_seconds=float(
            network_contract["station_refractory_seconds"]
        ),
        network_refractory_seconds=float(
            network_contract["network_refractory_seconds"]
        ),
        enforce_travel_time=True,
    )
    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (
            float(row["available_epoch"]),
            str(row["physical_station_id"]),
            int(row["packet_index"]),
            str(row["record_key"]),
        ),
    )
    by_candidate_id: dict[int, dict[str, Any]] = {}
    events = []
    transitions = []
    for row in ordered:
        transition = confirmer.process_candidate(
            str(row["physical_station_id"]),
            estimated_p_time=float(row["estimated_p_epoch"]),
            available_time=float(row["available_epoch"]),
            packet_index=int(row["packet_index"]),
        )
        candidate_id = int(transition["candidate"]["candidate_id"])
        by_candidate_id[candidate_id] = row
        transition_row = {
            "record_key": row["record_key"],
            "candidate_id": candidate_id,
            "physical_station_id": row["physical_station_id"],
            "available_epoch": float(row["available_epoch"]),
            "action": transition["action"],
            "network_trigger": bool(transition["network_trigger"]),
            "physical_rejection_count": len(transition["physical_rejections"]),
        }
        transitions.append(transition_row)
        if not transition["network_trigger"]:
            continue
        event = transition["network_event"]
        contributing = [by_candidate_id[int(value)] for value in event["candidate_ids"]]
        match_flags = [
            parse_bool(candidate.get("pick_match", False)) for candidate in contributing
        ]
        events.append(
            {
                "network_event_id": int(event["network_event_id"]),
                "physical_station_ids": [
                    str(candidate["physical_station_id"]) for candidate in contributing
                ],
                "candidate_record_keys": [
                    str(candidate["record_key"]) for candidate in contributing
                ],
                "candidate_ids": [int(value) for value in event["candidate_ids"]],
                "estimated_p_start_epoch": float(event["estimated_p_start_time"]),
                "estimated_p_end_epoch": float(event["estimated_p_end_time"]),
                "estimated_p_span_seconds": float(event["estimated_p_span_seconds"]),
                "available_epoch": float(event["available_time"]),
                "matched_station_candidates": int(sum(match_flags)),
                "all_station_candidates_match": bool(all(match_flags)),
                "strict_false_network_confirmation": not bool(all(match_flags)),
                "travel_time": event["travel_time"],
            }
        )
    return events, transitions


__all__ = ["parse_bool", "replay_candidate_rows"]
