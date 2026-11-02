#!/usr/bin/env python3
"""Isolated, frozen-weight CSPP gate ablation on sealed missing-component inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
import gc
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/evaluation")]
import component_fault_experiment as previous
import cspp.evaluation.score_cache as core_cache
import cspp_v27_hybrid_score_cache as hybrid
from cspp.streaming.normalization import normalize_packet_ewma_causal
from cspp.streaming.quality import PacketQualityConfig, PacketQualityGate

PARENT = previous.DEFAULT
DEFAULT = ROOT / "outputs/cspp_component_gate_ablation_20260908"
VARIANTS = ("original_gate", "two_component_gate")


class TwoComponentDiagnosticGate(PacketQualityGate):
    """Accept one exact-zero channel only when it is the sole rejection reason."""

    def evaluate(self, packet, timestamp_seconds=None):
        prior_streak = self.valid_streak
        prior_timestamp = self.previous_timestamp
        decision = super().evaluate(packet, timestamp_seconds)
        if decision.reason_codes != ("constant_component",):
            return decision
        wave = np.asarray(packet)
        flat = np.ptp(wave, axis=1) == 0
        if int(flat.sum()) != 1 or np.count_nonzero(wave[flat]) != 0:
            return decision
        # super() has reset its state on rejection; restore only this accepted step.
        self.valid_streak = prior_streak + 1
        self.previous_packet = wave.copy()
        self.previous_timestamp = prior_timestamp if timestamp_seconds is None else float(timestamp_seconds)
        allowed = self.valid_streak >= self.config.warmup_packets
        return replace(decision, valid=True, candidate_allowed=allowed, reset_required=False,
                       reason_codes=(), valid_streak=self.valid_streak,
                       warmup_remaining=max(0, self.config.warmup_packets-self.valid_streak),
                       quality_flags=decision.quality_flags + ("one_zero_component_accepted",))


def normalize_and_gate(wave, *, start_epoch, packet_seconds, warmup_packets):
    """Same normalization and reset loop as the frozen scorer, with one new gate."""
    count = wave.shape[1] // 50
    normalized = np.zeros((count, 3, 50), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    allowed = np.zeros(count, dtype=bool)
    gate = TwoComponentDiagnosticGate(PacketQualityConfig(warmup_packets=int(warmup_packets)))
    state = None
    reasons, flags = Counter(), Counter()
    for index in range(count):
        packet = wave[:, index*50:(index+1)*50]
        decision = gate.evaluate(packet, float(start_epoch) + index*packet_seconds)
        reasons.update(decision.reason_codes)
        flags.update(decision.quality_flags)
        valid[index] = decision.valid
        allowed[index] = decision.candidate_allowed
        if not decision.valid:
            state = None
            continue
        normalized[index], state = normalize_packet_ewma_causal(packet, state)
    return normalized, valid, allowed, reasons, flags


def score(wave, record, protocol, ensemble, *, backend="hybrid", ablation=True):
    # Process-local injection avoids changing either the released code or its files.
    with ExitStack() as stack:
        if ablation:
            stack.enter_context(patch.object(core_cache, "_normalize_and_gate", normalize_and_gate))
            stack.enter_context(patch.object(hybrid, "_normalize_and_gate", normalize_and_gate))
        function = core_cache.score_waveform_packets if backend == "cpu" else hybrid.score_waveform_packets_hybrid
        return function(wave, start_epoch=record["start_epoch"], ensemble=ensemble, runtime=protocol["runtime"])


def prepare(output):
    parent, seal = previous.load_seal(PARENT)
    assert previous.read(PARENT / "verification.json")["passed"]
    sources = [Path(__file__), ROOT / "tests/test_component_gate_ablation.py", ROOT / "scripts/evaluation/aggregate_component_gate_ablation.m"]
    data = [PARENT / name for name in ("seal.json", "protocol.json", "verification.json", "cspp_complete.json", "record_metrics.csv")]
    if (output / "protocol.json").exists():
        load_protocol(output)
        return
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(), "role": "post-result diagnostic ablation; not a deployed fix or new blind test",
        "parent": str(PARENT), "sources": [previous.artifact(p) for p in sources],
        "parent_artifacts": [previous.artifact(p) for p in data],
        "input_source": "Reuse the exact 645 sealed four-condition HDF5 inputs without rewriting them",
        "variants": list(VARIANTS), "cases": previous.CASES, "event_records": 633, "events": 65, "background_station_hours": 12,
        "change": "Only waive constant_component rejection when exactly one channel is all zero, the other two are finite and nonconstant, and no other rejection reason exists",
        "unchanged": ["checkpoint and thresholds", "packet normalization, including zero channel in the original pooled scale", "features and candidate formation", "initial 20-packet warmup", "timestamp, nonfinite, duplicate-packet and multiple-flat-channel rejection"],
        "state_consequence": "An accepted missing-component packet preserves normalization/GRU/candidate history; no fault-specific reset or second warmup",
        "edge_audit": {"available_time_window_seconds": [0, 2], "estimated_time_distance_seconds": 0.5,
                       "boundaries": ["fault_start", "fault_end"], "background": "paired counts, not confirmed false alarms",
                       "event_caveat": "P-targeted fault placement means matched arrivals near the fault onset cannot alone prove recognition of the remaining waveform"},
        "runtime": parent["runtime"], "cspp_thresholds": parent["cspp_thresholds"], "model": parent["models"]["cspp"],
    }
    previous.write(output / "protocol.json", protocol)
    print("ablation protocol sealed before variant outputs", flush=True)


def load_protocol(output):
    protocol = previous.read(output / "protocol.json")
    for item in protocol["sources"] + protocol["parent_artifacts"]:
        previous.verify(item)
    parent, seal = previous.load_seal(Path(protocol["parent"]))
    return protocol, parent, seal


def pilot(output):
    _, parent, sealed = load_protocol(output)
    selected = parent["records"][:2]
    ensemble = previous.new_cspp(parent)
    cpu, comparisons = {}, {}
    for record in selected:
        for case, wave in enumerate(previous.load_waves(PARENT, record)):
            cpu[record["id"], case] = score(wave, record, parent, ensemble, backend="cpu")
            if case == 0:
                original = score(wave, record, parent, ensemble, backend="cpu", ablation=False)
                comparisons[record["id"] + ":clean_original"] = previous.compare_cache(cpu[record["id"], case], original)
    hybrid.prepare_hybrid_ensemble(ensemble)
    for record in selected:
        for case, wave in enumerate(previous.load_waves(PARENT, record)):
            cache = score(wave, record, parent, ensemble)
            comparisons[record["id"] + ":" + previous.CASES[case]] = previous.compare_cache(cpu[record["id"], case], cache)
            left = previous.cspp_candidates(cpu[record["id"], case], record, parent)
            right = previous.cspp_candidates(cache, record, parent)
            assert [c["episode_start_packet"] for c in left] == [c["episode_start_packet"] for c in right]
            for a, b in zip(left, right, strict=True):
                assert abs(a["estimated_p_epoch"] - b["estimated_p_epoch"]) < 1e-4
            changed = wave.copy()
            changed[:, 12000:] = 12345
            future = score(changed, record, parent, ensemble)
            for name, array in cache.arrays().items():
                assert np.allclose(array[:240], future.arrays()[name][:240], atol=1e-6, rtol=0, equal_nan=True)
        print(f"ablation pilot {record['id']}: CPU/hybrid, unchanged clean control, causality passed", flush=True)
    previous.write(output / "pilot.json", {"passed": True, "protocol_sha256": previous.sha256_file(output / "protocol.json"),
                                          "comparisons": comparisons})


def run(output):
    _, parent, sealed = load_protocol(output)
    check = previous.read(output / "pilot.json")
    protocol_hash = previous.sha256_file(output / "protocol.json")
    assert check["passed"] and check["protocol_sha256"] == protocol_hash
    torch.mps.set_per_process_memory_fraction(0.3)
    ensemble = previous.new_cspp(parent)
    hybrid.prepare_hybrid_ensemble(ensemble)
    source_specs = {s["path"]: s for s in sealed["inputs"]}
    for index, record in enumerate(parent["records"]):
        destination = output / "runs" / (record["id"] + ".json")
        source_spec = source_specs[str(previous.input_path(PARENT, record))]
        if destination.exists():
            prior = previous.read(destination)
            assert prior["protocol_sha256"] == protocol_hash and prior["input_sha256"] == source_spec["sha256"]
            continue
        previous.verify(source_spec)
        started = time.monotonic()
        results = []
        for case, wave in enumerate(previous.load_waves(PARENT, record)):
            start = time.monotonic()
            cache = score(wave, record, parent, ensemble)
            results.append({"case": previous.CASES[case], "candidates": previous.cspp_candidates(cache, record, parent),
                            "health": previous.health(cache, record, case), "wall_seconds": time.monotonic()-start})
        previous.write(destination, {"record_id": record["record_id"], "variant": VARIANTS[1], "results": results,
                                     "protocol_sha256": protocol_hash, "input_sha256": source_spec["sha256"]})
        if index % 20 == 0 or index >= 633:
            print(f"gate ablation {index+1}/645 {time.monotonic()-started:.2f}s", flush=True)
        gc.collect()
        torch.mps.empty_cache()
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8*2**30
        assert torch.mps.driver_allocated_memory() < 4*2**30
    previous.write(output / "complete.json", {"records": 645, "conditions": 4, "protocol_sha256": protocol_hash,
                                             "outputs": [previous.artifact(p) for p in sorted((output / "runs").glob("*.json"))]})


def export(output):
    _, parent, _ = load_protocol(output)
    old_complete = previous.read(PARENT / "cspp_complete.json")
    new_complete = previous.read(output / "complete.json")
    for complete in (old_complete, new_complete):
        assert len(complete["outputs"]) == 645
        for file in complete["outputs"]:
            previous.verify(file)
    candidates, health, boundaries = [], [], []
    for record in parent["records"]:
        for index, start in enumerate(record["fault_starts"]):
            boundaries.append({"id": record["id"], "fault_index": index, "start_epoch": record["start_epoch"] + start/100,
                               "end_epoch": record["start_epoch"] + start/100 + 10})
        for variant, path in ((VARIANTS[0], PARENT / "runs/cspp" / (record["id"] + ".json")),
                              (VARIANTS[1], output / "runs" / (record["id"] + ".json"))):
            data = previous.read(path)
            assert [r["case"] for r in data["results"]] == previous.CASES
            for r in data["results"]:
                base = {"id": record["id"], "kind": record["kind"], "event_id": record["event_id"],
                        "station_key": record["station_key"], "variant": variant, "case": r["case"],
                        "reference_p_epoch": record["reference_p_epoch"]}
                for candidate in r["candidates"]:
                    candidates.append({**base, "estimated_p_epoch": candidate["estimated_p_epoch"], "available_epoch": candidate["available_epoch"]})
                h = r["health"]
                recovery = h["recovery_after_fault_seconds"]
                health.append({**base, "candidate_count": len(r["candidates"]), "duration_seconds": record["sample_count"]/100,
                               "allowed_seconds": h["candidate_allowed_seconds"], "invalid_seconds": h["quality_invalid_seconds"],
                               "disallowed_during_fault_seconds": h["disallowed_during_fault_seconds"],
                               "recovery_mean_seconds": float(np.mean(recovery)) if all(v is not None for v in recovery) else None,
                               "accepted_zero_packets": h["diagnostics"]["quality_flags"].get("one_zero_component_accepted", 0)})
    previous.write_csv(output / "candidates.csv", candidates)
    previous.write_csv(output / "record_health.csv", health)
    previous.write_csv(output / "boundaries.csv", boundaries)
    previous.write(output / "export_complete.json", {"record_conditions": len(health), "candidate_count": len(candidates),
                    "files": [previous.artifact(output / n) for n in ("candidates.csv", "record_health.csv", "boundaries.csv")]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "run", "export"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
