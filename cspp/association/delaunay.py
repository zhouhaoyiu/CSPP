"""Bounded online Delaunay and travel-time confirmation for station picks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


EARTH_RADIUS_KM = 6371.0088


def edge_key(station_a: str, station_b: str) -> tuple[str, str]:
    first, second = sorted((str(station_a), str(station_b)))
    if first == second:
        raise ValueError("A Delaunay edge requires two stations")
    return first, second


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    lat_a, lon_a, lat_b, lon_b = map(
        math.radians,
        (latitude_a, longitude_a, latitude_b, longitude_b),
    )
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, value))))


@dataclass(frozen=True)
class NetworkCandidate:
    candidate_id: int
    station_id: str
    estimated_p_epoch: float
    available_epoch: float
    packet_index: int


class CausalDelaunayConfirmer:
    """Apply the first-three-trigger Delaunay sequence with bounded state.

    The state transition follows the interference-rejection logic described
    by Ma (2008, Section 2.8.4) and Ma et al. (2013). The implementation adds
    a predeclared pairwise travel-time necessary condition. It stores only the
    active sequence, one held suspect, and one refractory timestamp per
    station. Pair distances are evaluated on demand for the selected triple.
    """

    def __init__(
        self,
        station_coordinates: Mapping[str, Mapping[str, float]],
        delaunay_edges: Iterable[Iterable[str]],
        *,
        required_stations: int = 3,
        association_window_seconds: float = 10.0,
        minimum_apparent_velocity_km_s: float = 3.0,
        travel_time_tolerance_seconds: float = 1.0,
        station_refractory_seconds: float = 10.0,
    ):
        if required_stations != 3:
            raise ValueError("The Ma first-three sequence requires three stations")
        self.coordinates = {
            str(station): (
                float(values["latitude"]),
                float(values["longitude"]),
            )
            for station, values in station_coordinates.items()
        }
        if len(self.coordinates) < required_stations:
            raise ValueError("Network contains fewer than three stations")
        for station, (latitude, longitude) in self.coordinates.items():
            if (
                not math.isfinite(latitude)
                or not math.isfinite(longitude)
                or not -90.0 <= latitude <= 90.0
                or not -180.0 <= longitude <= 180.0
            ):
                raise ValueError(f"Invalid coordinates for {station}")
        self.edges = {
            edge_key(*tuple(str(value) for value in edge)) for edge in delaunay_edges
        }
        if not self.edges:
            raise ValueError("At least one Delaunay edge is required")
        unknown = {
            station
            for edge in self.edges
            for station in edge
            if station not in self.coordinates
        }
        if unknown:
            raise ValueError(f"Edges contain unknown stations: {sorted(unknown)}")
        for label, value, allow_zero in (
            (
                "association_window_seconds",
                association_window_seconds,
                False,
            ),
            (
                "minimum_apparent_velocity_km_s",
                minimum_apparent_velocity_km_s,
                False,
            ),
            (
                "travel_time_tolerance_seconds",
                travel_time_tolerance_seconds,
                True,
            ),
            (
                "station_refractory_seconds",
                station_refractory_seconds,
                True,
            ),
        ):
            if (
                not math.isfinite(value)
                or value < 0.0
                or (not allow_zero and value == 0.0)
            ):
                raise ValueError(f"{label} has an invalid value")
        self.required_stations = required_stations
        self.association_window_seconds = float(association_window_seconds)
        self.minimum_apparent_velocity_km_s = float(minimum_apparent_velocity_km_s)
        self.travel_time_tolerance_seconds = float(travel_time_tolerance_seconds)
        self.station_refractory_seconds = float(station_refractory_seconds)
        self._chain: list[NetworkCandidate] = []
        self._suspect: NetworkCandidate | None = None
        self._last_station_epoch: dict[str, float] = {}
        self._next_candidate_id = 1
        self._next_event_id = 1

    @property
    def active_state_candidates(self) -> int:
        return len(self._chain) + int(self._suspect is not None)

    def reset(self, *, clear_refractory: bool = True) -> None:
        self._chain.clear()
        self._suspect = None
        if clear_refractory:
            self._last_station_epoch.clear()

    def _adjacent(self, first: NetworkCandidate, second: NetworkCandidate) -> bool:
        return edge_key(first.station_id, second.station_id) in self.edges

    @staticmethod
    def _row(candidate: NetworkCandidate) -> dict:
        return {
            "candidate_id": candidate.candidate_id,
            "station_id": candidate.station_id,
            "estimated_p_epoch": candidate.estimated_p_epoch,
            "available_epoch": candidate.available_epoch,
            "packet_index": candidate.packet_index,
        }

    def _start(self, candidate: NetworkCandidate, action: str) -> dict:
        self._chain = [candidate]
        self._suspect = None
        return {
            "action": action,
            "accepted_candidate_ids": [candidate.candidate_id],
            "rejected_candidate_ids": [],
        }

    def _travel_time_audit(self, selected: list[NetworkCandidate]) -> dict:
        pair_rows = []
        all_compatible = True
        for index, first in enumerate(selected):
            for second in selected[index + 1 :]:
                lat_a, lon_a = self.coordinates[first.station_id]
                lat_b, lon_b = self.coordinates[second.station_id]
                distance = haversine_km(lat_a, lon_a, lat_b, lon_b)
                observed = abs(first.estimated_p_epoch - second.estimated_p_epoch)
                maximum = (
                    distance / self.minimum_apparent_velocity_km_s
                    + self.travel_time_tolerance_seconds
                )
                compatible = observed <= maximum
                all_compatible &= compatible
                pair_rows.append(
                    {
                        "station_a": first.station_id,
                        "station_b": second.station_id,
                        "distance_km": distance,
                        "observed_arrival_difference_seconds": observed,
                        "maximum_arrival_difference_seconds": maximum,
                        "compatible": bool(compatible),
                    }
                )
        return {
            "all_pairs_compatible": bool(all_compatible),
            "minimum_apparent_velocity_km_s": (self.minimum_apparent_velocity_km_s),
            "tolerance_seconds": self.travel_time_tolerance_seconds,
            "pairs": pair_rows,
        }

    def _maybe_emit(self) -> tuple[dict | None, dict | None]:
        if len(self._chain) < self.required_stations:
            return None, None
        selected = self._chain[-self.required_stations :]
        span = max(value.estimated_p_epoch for value in selected) - min(
            value.estimated_p_epoch for value in selected
        )
        if span > self.association_window_seconds:
            return None, {
                "reason": "estimated_p_span_exceeds_window",
                "estimated_p_span_seconds": span,
            }
        travel_time = self._travel_time_audit(selected)
        if not travel_time["all_pairs_compatible"]:
            return None, {
                "reason": "pairwise_travel_time_incompatible",
                "estimated_p_span_seconds": span,
                "travel_time": travel_time,
            }
        event = {
            "network_event_id": self._next_event_id,
            "stations": [value.station_id for value in selected],
            "candidate_ids": [value.candidate_id for value in selected],
            "estimated_p_start_epoch": min(
                value.estimated_p_epoch for value in selected
            ),
            "estimated_p_end_epoch": max(value.estimated_p_epoch for value in selected),
            "estimated_p_span_seconds": span,
            "available_epoch": max(value.available_epoch for value in selected),
            "travel_time": travel_time,
        }
        self._next_event_id += 1
        self._chain.clear()
        self._suspect = None
        return event, None

    def process(
        self,
        station_id: str,
        *,
        estimated_p_epoch: float,
        available_epoch: float,
        packet_index: int,
    ) -> dict:
        station = str(station_id)
        if station not in self.coordinates:
            raise ValueError(f"Unknown station: {station}")
        if not math.isfinite(estimated_p_epoch) or not math.isfinite(available_epoch):
            raise ValueError("Candidate times must be finite")
        candidate = NetworkCandidate(
            candidate_id=self._next_candidate_id,
            station_id=station,
            estimated_p_epoch=float(estimated_p_epoch),
            available_epoch=float(available_epoch),
            packet_index=int(packet_index),
        )
        self._next_candidate_id += 1
        previous = self._last_station_epoch.get(station)
        self._last_station_epoch[station] = candidate.available_epoch
        if (
            previous is not None
            and candidate.available_epoch - previous <= self.station_refractory_seconds
        ):
            transition = {
                "action": "rejected_station_refractory",
                "accepted_candidate_ids": [],
                "rejected_candidate_ids": [candidate.candidate_id],
            }
        else:
            active = [
                *self._chain,
                *([] if self._suspect is None else [self._suspect]),
            ]
            if active and (
                candidate.available_epoch
                - min(value.available_epoch for value in active)
                > self.association_window_seconds
            ):
                transition = self._start(candidate, "window_expired_restart")
            elif not self._chain:
                transition = self._start(candidate, "accepted_as_first")
            elif candidate.station_id in {value.station_id for value in active}:
                transition = {
                    "action": "rejected_duplicate_station_in_sequence",
                    "accepted_candidate_ids": [],
                    "rejected_candidate_ids": [candidate.candidate_id],
                }
            elif self._suspect is None:
                anchor = self._chain[-1]
                if self._adjacent(anchor, candidate):
                    self._chain.append(candidate)
                    transition = {
                        "action": "accepted_adjacent_next",
                        "accepted_candidate_ids": [candidate.candidate_id],
                        "rejected_candidate_ids": [],
                    }
                else:
                    self._suspect = candidate
                    transition = {
                        "action": "held_nonadjacent_for_third",
                        "accepted_candidate_ids": [],
                        "rejected_candidate_ids": [],
                    }
            else:
                anchor = self._chain[-1]
                suspect = self._suspect
                if self._adjacent(anchor, candidate):
                    self._chain.append(candidate)
                    self._suspect = None
                    transition = {
                        "action": "rejected_nonadjacent_second",
                        "accepted_candidate_ids": [candidate.candidate_id],
                        "rejected_candidate_ids": [suspect.candidate_id],
                    }
                elif self._adjacent(suspect, candidate):
                    rejected = [value.candidate_id for value in self._chain]
                    self._chain = [suspect, candidate]
                    self._suspect = None
                    transition = {
                        "action": "rejected_original_first",
                        "accepted_candidate_ids": [
                            suspect.candidate_id,
                            candidate.candidate_id,
                        ],
                        "rejected_candidate_ids": rejected,
                    }
                else:
                    rejected = [
                        *[value.candidate_id for value in self._chain],
                        suspect.candidate_id,
                    ]
                    transition = self._start(candidate, "all_three_unrelated_restart")
                    transition["rejected_candidate_ids"] = rejected
        event, physical_rejection = self._maybe_emit()
        return {
            "candidate": self._row(candidate),
            **transition,
            "network_trigger": event is not None,
            "network_event": event,
            "physical_rejection": physical_rejection,
            "active_state_candidates": self.active_state_candidates,
        }


def self_test() -> None:
    coordinates = {
        "A": {"latitude": 0.0, "longitude": 0.00},
        "B": {"latitude": 0.0, "longitude": 0.05},
        "C": {"latitude": 0.0, "longitude": 0.10},
        "D": {"latitude": 0.0, "longitude": 0.15},
    }
    confirmer = CausalDelaunayConfirmer(
        coordinates,
        [("A", "B"), ("B", "C"), ("C", "D")],
        station_refractory_seconds=0.0,
    )
    confirmer.process(
        "A", estimated_p_epoch=10.0, available_epoch=10.5, packet_index=20
    )
    confirmer.process(
        "B", estimated_p_epoch=10.5, available_epoch=11.0, packet_index=21
    )
    result = confirmer.process(
        "C", estimated_p_epoch=11.0, available_epoch=11.5, packet_index=22
    )
    assert result["network_trigger"]
    assert result["network_event"]["stations"] == ["A", "B", "C"]
    assert result["active_state_candidates"] == 0

    confirmer.reset()
    confirmer.process(
        "A", estimated_p_epoch=20.0, available_epoch=20.5, packet_index=40
    )
    held = confirmer.process(
        "C", estimated_p_epoch=20.4, available_epoch=21.0, packet_index=41
    )
    assert held["action"] == "held_nonadjacent_for_third"
    repaired = confirmer.process(
        "B", estimated_p_epoch=20.7, available_epoch=21.5, packet_index=42
    )
    assert repaired["action"] == "rejected_nonadjacent_second"
    assert not repaired["network_trigger"]
    result = confirmer.process(
        "C", estimated_p_epoch=21.0, available_epoch=22.0, packet_index=43
    )
    assert result["network_trigger"]

    before = confirmer.active_state_candidates
    for index in range(100):
        confirmer.process(
            "D",
            estimated_p_epoch=100.0 + index,
            available_epoch=100.5 + index,
            packet_index=200 + index,
        )
    assert confirmer.active_state_candidates <= 2
    assert before <= 2
    print("Causal Delaunay association self-test: PASS")


__all__ = [
    "CausalDelaunayConfirmer",
    "NetworkCandidate",
    "edge_key",
    "haversine_km",
    "self_test",
]
