"""Bounded connected-triplet and travel-time network confirmation.

The Delaunay-neighbor screen is motivated by Ma et al. (2013),
doi:10.6038/cjg20130718. This implementation is an extension rather than a
reproduction: it requires a connected three-station subgraph and additionally
checks a pairwise constant-velocity travel-time necessary condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, Mapping


EARTH_RADIUS_KM = 6371.0088


def _edge_key(station_a: str, station_b: str) -> tuple[str, str]:
    if station_a == station_b:
        raise ValueError("A Delaunay edge must join different stations")
    return tuple(sorted((station_a, station_b)))


def _haversine_km(
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
class StationCandidate:
    candidate_id: int
    station_id: str
    estimated_p_time: float
    available_time: float
    packet_index: int


class ConnectedTripletTravelTimeConfirmer:
    """Confirm a connected Delaunay triplet with bounded online state.

    One latest candidate is retained per station inside the availability
    window. A new candidate can complete only triplets in its two-hop Delaunay
    neighborhood. Every retained pair must satisfy the declared constant-
    velocity necessary condition. The best valid triplet is selected
    deterministically by estimated-P span and station code.
    """

    def __init__(
        self,
        station_coordinates: Mapping[str, Mapping[str, float]],
        edges: Iterable[Iterable[str]],
        *,
        required_stations: int = 3,
        association_window_seconds: float = 10.0,
        minimum_p_velocity_km_s: float = 4.0,
        travel_time_tolerance_seconds: float = 1.0,
        station_refractory_seconds: float = 10.0,
        network_refractory_seconds: float = 10.0,
        enforce_travel_time: bool = True,
    ):
        if required_stations != 3:
            raise ValueError("Connected-triplet confirmation requires three stations")
        coordinates: dict[str, tuple[float, float]] = {}
        for station_id, values in station_coordinates.items():
            station = str(station_id)
            latitude = float(values["latitude"])
            longitude = float(values["longitude"])
            if not math.isfinite(latitude) or not math.isfinite(longitude):
                raise ValueError(f"Station {station!r} has non-finite coordinates")
            if not -90.0 <= latitude <= 90.0:
                raise ValueError(f"Station {station!r} latitude is out of range")
            if not -180.0 <= longitude <= 180.0:
                raise ValueError(f"Station {station!r} longitude is out of range")
            coordinates[station] = (latitude, longitude)
        if len(coordinates) < required_stations:
            raise ValueError("Network has fewer stations than required")
        for name, value, allow_zero in (
            ("association_window_seconds", association_window_seconds, False),
            ("minimum_p_velocity_km_s", minimum_p_velocity_km_s, False),
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
            (
                "network_refractory_seconds",
                network_refractory_seconds,
                True,
            ),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or (
                numeric < 0.0 if allow_zero else numeric <= 0.0
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}")

        edge_set = {_edge_key(*tuple(str(value) for value in edge)) for edge in edges}
        unknown = {
            station
            for edge in edge_set
            for station in edge
            if station not in coordinates
        }
        if unknown:
            raise ValueError(f"Edges reference unknown stations: {sorted(unknown)}")
        if not edge_set:
            raise ValueError("At least one Delaunay edge is required")
        adjacency = {station: set() for station in coordinates}
        for station_a, station_b in edge_set:
            adjacency[station_a].add(station_b)
            adjacency[station_b].add(station_a)

        self.station_coordinates = coordinates
        self.edges = edge_set
        self.adjacency = adjacency
        self.required_stations = required_stations
        self.association_window_seconds = float(association_window_seconds)
        self.minimum_p_velocity_km_s = float(minimum_p_velocity_km_s)
        self.travel_time_tolerance_seconds = float(travel_time_tolerance_seconds)
        self.station_refractory_seconds = float(station_refractory_seconds)
        self.network_refractory_seconds = float(network_refractory_seconds)
        self.enforce_travel_time = bool(enforce_travel_time)
        self._distance_cache: dict[tuple[str, str], float] = {}
        self._active_by_station: dict[str, StationCandidate] = {}
        self._last_station_candidate_time: dict[str, float] = {}
        self._last_available_time: float | None = None
        self._last_network_event_time: float | None = None
        self._next_candidate_id = 1
        self._next_network_event_id = 1

    def reset(self) -> None:
        self._active_by_station.clear()
        self._last_station_candidate_time.clear()
        self._last_available_time = None
        self._last_network_event_time = None

    def _candidate_row(self, candidate: StationCandidate) -> dict:
        return {
            "candidate_id": candidate.candidate_id,
            "station_id": candidate.station_id,
            "estimated_p_time": candidate.estimated_p_time,
            "available_time": candidate.available_time,
            "packet_index": candidate.packet_index,
        }

    def _distance_km(self, station_a: str, station_b: str) -> float:
        key = _edge_key(station_a, station_b)
        if key not in self._distance_cache:
            latitude_a, longitude_a = self.station_coordinates[key[0]]
            latitude_b, longitude_b = self.station_coordinates[key[1]]
            self._distance_cache[key] = _haversine_km(
                latitude_a,
                longitude_a,
                latitude_b,
                longitude_b,
            )
        return self._distance_cache[key]

    def _is_connected_triplet(
        self,
        candidates: tuple[StationCandidate, StationCandidate, StationCandidate],
    ) -> bool:
        stations = [candidate.station_id for candidate in candidates]
        edge_count = sum(
            _edge_key(station_a, station_b) in self.edges
            for station_a, station_b in combinations(stations, 2)
        )
        return edge_count >= 2

    def _travel_time_audit(
        self,
        candidates: tuple[StationCandidate, StationCandidate, StationCandidate],
    ) -> dict:
        pair_rows = []
        all_compatible = True
        maximum_fraction = 0.0
        for first, second in combinations(candidates, 2):
            distance_km = self._distance_km(
                first.station_id,
                second.station_id,
            )
            observed = abs(first.estimated_p_time - second.estimated_p_time)
            maximum = (
                distance_km / self.minimum_p_velocity_km_s
                + self.travel_time_tolerance_seconds
            )
            compatible = observed <= maximum
            all_compatible &= compatible
            fraction = observed / maximum if maximum else math.inf
            maximum_fraction = max(maximum_fraction, fraction)
            pair_rows.append(
                {
                    "station_a": first.station_id,
                    "station_b": second.station_id,
                    "distance_km": distance_km,
                    "observed_arrival_difference_seconds": observed,
                    "maximum_arrival_difference_seconds": maximum,
                    "fraction_of_maximum": fraction,
                    "compatible": compatible,
                }
            )
        return {
            "enforced": self.enforce_travel_time,
            "all_pairs_compatible": bool(all_compatible),
            "maximum_fraction_of_allowed_difference": maximum_fraction,
            "minimum_p_velocity_km_s": self.minimum_p_velocity_km_s,
            "tolerance_seconds": self.travel_time_tolerance_seconds,
            "pairs": pair_rows,
        }

    def _local_active_candidates(
        self,
        station: str,
    ) -> list[StationCandidate]:
        one_hop = self.adjacency[station]
        two_hop = set(one_hop)
        for neighbor in one_hop:
            two_hop.update(self.adjacency[neighbor])
        two_hop.add(station)
        return [
            self._active_by_station[value]
            for value in sorted(two_hop)
            if value in self._active_by_station
        ]

    def _candidate_triplets(
        self,
        newest: StationCandidate,
    ) -> tuple[list[dict], list[dict]]:
        local = [
            candidate
            for candidate in self._local_active_candidates(newest.station_id)
            if candidate.station_id != newest.station_id
        ]
        valid = []
        physical_rejections = []
        for first, second in combinations(local, 2):
            candidates = (first, second, newest)
            if not self._is_connected_triplet(candidates):
                continue
            estimated_times = [candidate.estimated_p_time for candidate in candidates]
            span = max(estimated_times) - min(estimated_times)
            if span > self.association_window_seconds:
                continue
            travel_time = self._travel_time_audit(candidates)
            if self.enforce_travel_time and not travel_time["all_pairs_compatible"]:
                physical_rejections.append(
                    {
                        "candidate_ids": [
                            candidate.candidate_id for candidate in candidates
                        ],
                        "stations": [candidate.station_id for candidate in candidates],
                        "estimated_p_span_seconds": span,
                        "travel_time": travel_time,
                    }
                )
                continue
            stations = tuple(sorted(candidate.station_id for candidate in candidates))
            valid.append(
                {
                    "candidates": candidates,
                    "estimated_p_span_seconds": span,
                    "travel_time": travel_time,
                    "score": (
                        span,
                        travel_time["maximum_fraction_of_allowed_difference"],
                        stations,
                    ),
                }
            )
        return valid, physical_rejections

    def process_candidate(
        self,
        station_id: str,
        *,
        estimated_p_time: float,
        available_time: float,
        packet_index: int,
    ) -> dict:
        station = str(station_id)
        if station not in self.station_coordinates:
            raise ValueError(f"Station {station!r} is not in the network config")
        if not math.isfinite(estimated_p_time):
            raise ValueError("estimated_p_time must be finite")
        if not math.isfinite(available_time):
            raise ValueError("available_time must be finite")
        if (
            self._last_available_time is not None
            and available_time < self._last_available_time
        ):
            raise ValueError("Candidates must be processed by available_time")
        self._last_available_time = float(available_time)
        candidate = StationCandidate(
            candidate_id=self._next_candidate_id,
            station_id=station,
            estimated_p_time=float(estimated_p_time),
            available_time=float(available_time),
            packet_index=int(packet_index),
        )
        self._next_candidate_id += 1

        cutoff = available_time - self.association_window_seconds
        expired = [
            value
            for value, active in self._active_by_station.items()
            if active.available_time < cutoff
        ]
        for value in expired:
            del self._active_by_station[value]

        if (
            self._last_network_event_time is not None
            and available_time - self._last_network_event_time
            <= self.network_refractory_seconds
        ):
            return {
                "candidate": self._candidate_row(candidate),
                "action": "rejected_network_refractory",
                "network_trigger": False,
                "network_event": None,
                "physical_rejections": [],
                "active_station_count": len(self._active_by_station),
            }

        previous_station_time = self._last_station_candidate_time.get(station)
        self._last_station_candidate_time[station] = candidate.available_time
        if (
            previous_station_time is not None
            and candidate.available_time - previous_station_time
            <= self.station_refractory_seconds
        ):
            return {
                "candidate": self._candidate_row(candidate),
                "action": "rejected_station_refractory",
                "network_trigger": False,
                "network_event": None,
                "physical_rejections": [],
                "active_station_count": len(self._active_by_station),
            }

        self._active_by_station[station] = candidate
        valid, physical_rejections = self._candidate_triplets(candidate)
        if not valid:
            return {
                "candidate": self._candidate_row(candidate),
                "action": (
                    "held_after_physical_rejection"
                    if physical_rejections
                    else "accepted_waiting_for_connected_triplet"
                ),
                "network_trigger": False,
                "network_event": None,
                "physical_rejections": physical_rejections,
                "active_station_count": len(self._active_by_station),
            }

        selected = min(valid, key=lambda value: value["score"])
        candidates = selected["candidates"]
        network_event_id = self._next_network_event_id
        self._next_network_event_id += 1
        event = {
            "network_event_id": network_event_id,
            "stations": [candidate.station_id for candidate in candidates],
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "estimated_p_start_time": min(
                candidate.estimated_p_time for candidate in candidates
            ),
            "estimated_p_end_time": max(
                candidate.estimated_p_time for candidate in candidates
            ),
            "estimated_p_span_seconds": selected["estimated_p_span_seconds"],
            "available_time": max(candidate.available_time for candidate in candidates),
            "travel_time": selected["travel_time"],
            "rule": {
                "connected_delaunay_triplet": True,
                "required_stations": self.required_stations,
                "association_window_seconds": self.association_window_seconds,
                "pairwise_travel_time_necessary_condition": (self.enforce_travel_time),
                "deterministic_selection": (
                    "minimum estimated-P span, then minimum maximum "
                    "travel-time fraction, then sorted station codes"
                ),
            },
        }
        self._active_by_station.clear()
        self._last_network_event_time = float(event["available_time"])
        return {
            "candidate": self._candidate_row(candidate),
            "action": "connected_triplet_confirmed",
            "network_trigger": True,
            "network_event": event,
            "physical_rejections": physical_rejections,
            "active_station_count": 0,
        }


def self_test() -> None:
    coordinates = {
        "A": {"latitude": 0.0, "longitude": 0.00},
        "B": {"latitude": 0.0, "longitude": 0.05},
        "C": {"latitude": 0.0, "longitude": 0.10},
        "D": {"latitude": 0.0, "longitude": 0.15},
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    confirmer = ConnectedTripletTravelTimeConfirmer(
        coordinates,
        edges,
        station_refractory_seconds=0.0,
        network_refractory_seconds=0.0,
    )
    confirmer.process_candidate(
        "A",
        estimated_p_time=10.0,
        available_time=10.5,
        packet_index=20,
    )
    confirmer.process_candidate(
        "C",
        estimated_p_time=11.0,
        available_time=11.0,
        packet_index=21,
    )
    result = confirmer.process_candidate(
        "B",
        estimated_p_time=10.5,
        available_time=11.5,
        packet_index=22,
    )
    assert result["network_trigger"]
    assert set(result["network_event"]["stations"]) == {"A", "B", "C"}

    confirmer.reset()
    confirmer.process_candidate(
        "A",
        estimated_p_time=20.0,
        available_time=20.5,
        packet_index=40,
    )
    confirmer.process_candidate(
        "B",
        estimated_p_time=28.0,
        available_time=28.5,
        packet_index=56,
    )
    rejected = confirmer.process_candidate(
        "C",
        estimated_p_time=28.5,
        available_time=29.0,
        packet_index=57,
    )
    assert not rejected["network_trigger"]
    assert rejected["physical_rejections"]

    false_coordinates = {
        "LMS": {"latitude": 33.81744, "longitude": -117.44500},
        "LMH": {"latitude": 33.74418, "longitude": -117.39135},
        "ELS2": {"latitude": 33.64907, "longitude": -117.42604},
    }
    false_confirmer = ConnectedTripletTravelTimeConfirmer(
        false_coordinates,
        [("LMS", "LMH"), ("LMH", "ELS2")],
        station_refractory_seconds=0.0,
        network_refractory_seconds=0.0,
    )
    false_confirmer.process_candidate(
        "LMS",
        estimated_p_time=1780035264.589244,
        available_time=1780035265.5,
        packet_index=1,
    )
    false_confirmer.process_candidate(
        "LMH",
        estimated_p_time=1780035270.331320,
        available_time=1780035271.5,
        packet_index=2,
    )
    false_result = false_confirmer.process_candidate(
        "ELS2",
        estimated_p_time=1780035270.952950,
        available_time=1780035272.0,
        packet_index=3,
    )
    assert not false_result["network_trigger"]
    assert false_result["physical_rejections"]
    print("V10.9 connected-triplet travel-time self-test: PASS")


if __name__ == "__main__":
    self_test()


__all__ = [
    "ConnectedTripletTravelTimeConfirmer",
    "StationCandidate",
    "self_test",
]
