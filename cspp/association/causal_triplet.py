"""Causal order-invariant connected-triplet association for V18."""

from __future__ import annotations

from itertools import permutations
from typing import Iterable, Mapping

from cspp.association.connected_triplet import ConnectedTripletTravelTimeConfirmer


class CausalConnectedTripletConfirmer(ConnectedTripletTravelTimeConfirmer):
    """Expose the connected-triplet rule through the V17 epoch API.

    The scientific rule remains the frozen V10.9 connected-triplet rule:
    retain at most one candidate per configured station, require a connected
    three-station Delaunay subgraph, and enforce pairwise travel-time
    compatibility. This adapter normalizes field names and event ordering for
    deterministic V18 evaluation.
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
        network_refractory_seconds: float = 10.0,
    ):
        super().__init__(
            station_coordinates,
            delaunay_edges,
            required_stations=required_stations,
            association_window_seconds=association_window_seconds,
            minimum_p_velocity_km_s=minimum_apparent_velocity_km_s,
            travel_time_tolerance_seconds=(travel_time_tolerance_seconds),
            station_refractory_seconds=station_refractory_seconds,
            network_refractory_seconds=network_refractory_seconds,
            enforce_travel_time=True,
        )

    @property
    def active_state_candidates(self) -> int:
        return len(self._active_by_station)

    @property
    def maximum_active_state_candidates(self) -> int:
        return len(self.station_coordinates)

    def process(
        self,
        station_id: str,
        *,
        estimated_p_epoch: float,
        available_epoch: float,
        packet_index: int,
    ) -> dict:
        transition = self.process_candidate(
            station_id,
            estimated_p_time=estimated_p_epoch,
            available_time=available_epoch,
            packet_index=packet_index,
        )
        candidate = transition["candidate"]
        normalized = {
            "candidate_id": int(candidate["candidate_id"]),
            "station_id": str(candidate["station_id"]),
            "estimated_p_epoch": float(candidate["estimated_p_time"]),
            "available_epoch": float(candidate["available_time"]),
            "packet_index": int(candidate["packet_index"]),
        }
        event = transition["network_event"]
        if event is not None:
            event = dict(event)
            station_candidates = sorted(
                zip(
                    event["stations"],
                    event["candidate_ids"],
                    strict=True,
                )
            )
            event["stations"] = [str(station) for station, _ in station_candidates]
            event["candidate_ids"] = [
                int(candidate_id) for _, candidate_id in station_candidates
            ]
            event["estimated_p_start_epoch"] = float(
                event.pop("estimated_p_start_time")
            )
            event["estimated_p_end_epoch"] = float(event.pop("estimated_p_end_time"))
            event["available_epoch"] = float(event.pop("available_time"))
        physical_rejections = transition["physical_rejections"]
        return {
            **transition,
            "candidate": normalized,
            "network_event": event,
            "physical_rejection": (
                physical_rejections[0] if physical_rejections else None
            ),
            "active_state_candidates": self.active_state_candidates,
            "maximum_active_state_candidates": (self.maximum_active_state_candidates),
        }


def _feed_order(
    order: tuple[str, ...],
) -> tuple[bool, tuple[str, ...], int]:
    coordinates = {
        "A": {"latitude": 0.0, "longitude": 0.00},
        "B": {"latitude": 0.0, "longitude": 0.05},
        "C": {"latitude": 0.0, "longitude": 0.10},
    }
    confirmer = CausalConnectedTripletConfirmer(
        coordinates,
        [("A", "B"), ("A", "C")],
        station_refractory_seconds=0.0,
        network_refractory_seconds=0.0,
    )
    estimated = {"A": 10.0, "B": 10.4, "C": 10.8}
    result = None
    for index, station in enumerate(order):
        result = confirmer.process(
            station,
            estimated_p_epoch=estimated[station],
            available_epoch=20.0 + index,
            packet_index=100 + index,
        )
    assert result is not None
    event = result["network_event"]
    stations = tuple(event["stations"]) if event is not None else ()
    return (
        bool(result["network_trigger"]),
        stations,
        int(result["active_state_candidates"]),
    )


def self_test() -> None:
    for order in permutations(("A", "B", "C")):
        triggered, stations, active = _feed_order(order)
        assert triggered
        assert stations == ("A", "B", "C")
        assert active == 0

    coordinates = {
        "A": {"latitude": 0.0, "longitude": 0.00},
        "B": {"latitude": 0.0, "longitude": 0.05},
        "C": {"latitude": 0.0, "longitude": 0.10},
        "D": {"latitude": 0.0, "longitude": 0.15},
    }
    confirmer = CausalConnectedTripletConfirmer(
        coordinates,
        [("A", "B"), ("B", "C"), ("C", "D")],
        station_refractory_seconds=0.0,
        network_refractory_seconds=0.0,
    )
    confirmer.process(
        "A",
        estimated_p_epoch=30.0,
        available_epoch=30.5,
        packet_index=1,
    )
    confirmer.process(
        "B",
        estimated_p_epoch=38.0,
        available_epoch=38.5,
        packet_index=2,
    )
    rejected = confirmer.process(
        "C",
        estimated_p_epoch=38.5,
        available_epoch=39.0,
        packet_index=3,
    )
    assert not rejected["network_trigger"]
    assert rejected["physical_rejection"] is not None

    confirmer.reset()
    confirmer.process(
        "A",
        estimated_p_epoch=50.0,
        available_epoch=50.5,
        packet_index=4,
    )
    confirmer.process(
        "B",
        estimated_p_epoch=50.4,
        available_epoch=50.9,
        packet_index=5,
    )
    expired = confirmer.process(
        "C",
        estimated_p_epoch=50.8,
        available_epoch=61.0,
        packet_index=6,
    )
    assert not expired["network_trigger"]
    assert expired["active_state_candidates"] == 1

    refractory = CausalConnectedTripletConfirmer(
        coordinates,
        [("A", "B"), ("B", "C"), ("C", "D")],
        station_refractory_seconds=10.0,
        network_refractory_seconds=10.0,
    )
    refractory.process(
        "A",
        estimated_p_epoch=70.0,
        available_epoch=70.5,
        packet_index=7,
    )
    duplicate = refractory.process(
        "A",
        estimated_p_epoch=70.2,
        available_epoch=71.0,
        packet_index=8,
    )
    assert duplicate["action"] == "rejected_station_refractory"
    refractory.process(
        "B",
        estimated_p_epoch=70.4,
        available_epoch=71.5,
        packet_index=9,
    )
    event = refractory.process(
        "C",
        estimated_p_epoch=70.8,
        available_epoch=72.0,
        packet_index=10,
    )
    assert event["network_trigger"]
    assert "estimated_p_start_time" not in event["network_event"]
    assert "estimated_p_start_epoch" in event["network_event"]
    blocked = refractory.process(
        "D",
        estimated_p_epoch=71.0,
        available_epoch=72.5,
        packet_index=11,
    )
    assert blocked["action"] == "rejected_network_refractory"

    bounded = CausalConnectedTripletConfirmer(
        coordinates,
        [("A", "B"), ("B", "C"), ("C", "D")],
        station_refractory_seconds=0.0,
        network_refractory_seconds=0.0,
    )
    for index in range(1000):
        station = ("A", "D")[index % 2]
        result = bounded.process(
            station,
            estimated_p_epoch=100.0 + index,
            available_epoch=100.5 + index,
            packet_index=100 + index,
        )
        assert (
            result["active_state_candidates"]
            <= result["maximum_active_state_candidates"]
        )
    assert bounded.active_state_candidates <= 2
    print("Causal connected-triplet association self-test: PASS")


if __name__ == "__main__":
    self_test()


__all__ = [
    "CausalConnectedTripletConfirmer",
    "self_test",
]
