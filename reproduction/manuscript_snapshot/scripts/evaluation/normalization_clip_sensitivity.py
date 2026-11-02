#!/usr/bin/env python3
"""Frozen-weight, paired sensitivity to the normalized input clipping limit."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from functools import partial
import gc
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).parent)]
import component_fault_experiment as base
import persistent_component_missing as previous
import cspp.evaluation.score_cache as core
import cspp_v27_hybrid_score_cache as hybrid
from cspp.streaming.normalization import normalize_packet_ewma_causal
from cspp.streaming.quality import PacketQualityConfig, PacketQualityGate

DEFAULT = ROOT / "outputs/cspp_normalization_clip_20260909"
LIMITS = {"clip6": 6.0, "clip12": 12.0, "clip24": 24.0, "unlimited": np.inf}
ORIGINAL_LOOP = core._normalize_and_gate


def load_wave(record):
    base.verify(record["clean_input"])
    with h5py.File(record["clean_input"]["path"]) as handle:
        wave = handle["wave"][:]
    assert wave.dtype == np.float32 and wave.shape == (3, record["sample_count"])
    return wave


def normalized_loop(wave, *, limit, **kwargs):
    with patch.object(core, "normalize_packet_ewma_causal", partial(normalize_packet_ewma_causal, clip=limit)):
        return ORIGINAL_LOOP(wave, **kwargs)


def score(wave, record, protocol, ensemble, case, backend="hybrid"):
    loop = partial(normalized_loop, limit=LIMITS[case])
    with ExitStack() as stack:
        stack.enter_context(patch.object(core, "_normalize_and_gate", loop))
        stack.enter_context(patch.object(hybrid, "_normalize_and_gate", loop))
        function = core.score_waveform_packets if backend == "cpu" else hybrid.score_waveform_packets_hybrid
        return function(wave, start_epoch=record["start_epoch"], ensemble=ensemble, runtime=protocol["runtime"])


def clipping_audit(wave, record, runtime):
    """Count actual float64 pre-clip exceedances, independent of model features."""
    gate = PacketQualityGate(PacketQualityConfig(warmup_packets=runtime["warmup_packets"]))
    state = None
    count = wave.shape[1] // 50
    valid = np.zeros(count, dtype=bool)
    allowed = valid.copy()
    raw_values, sample_indices = [], []
    for index in range(count):
        packet = wave[:, index*50:(index+1)*50]
        decision = gate.evaluate(packet, record["start_epoch"] + index*0.5)
        valid[index], allowed[index] = decision.valid, decision.candidate_allowed
        if not decision.valid:
            state = None
            continue
        values, next_state = normalize_packet_ewma_causal(packet, state, clip=np.inf)
        energy = state["rms_sq"] if state is not None else next_state["rms_sq"]
        packet64 = packet.astype(np.float64)
        before = (packet64-packet64.mean(axis=1, keepdims=True))/(np.sqrt(energy)+1e-8)
        assert np.array_equal(before.astype(np.float32), values)
        raw_values.append(before)
        sample_indices.append(np.arange(index*50, (index+1)*50))
        state = next_state
    raw = np.stack(raw_values) if raw_values else np.empty((0, 3, 50))
    times = np.concatenate(sample_indices)/100 if sample_indices else np.empty(0)
    flattened = raw.transpose(1, 0, 2).reshape(3, -1)
    scopes = {"all_valid": np.ones(len(times), dtype=bool)}
    if record["kind"] == "event":
        relative_p = record["reference_p_epoch"]-record["start_epoch"]
        scopes["p_0_to_1s"] = (times >= relative_p) & (times < relative_p+1)
        scopes["pre_p_1s"] = (times >= relative_p-1) & (times < relative_p)
    results = []
    for case, limit in LIMITS.items():
        for scope, mask in scopes.items():
            values = flattened[:, mask]
            exceeded = np.abs(values) > limit
            results.append({"case": case, "scope": scope, "valid_samples_per_axis": int(mask.sum()),
                            "clipped_Z": int(exceeded[0].sum()), "clipped_N": int(exceeded[1].sum()),
                            "clipped_E": int(exceeded[2].sum()), "clipped_total": int(exceeded.sum()),
                            "sample_total": int(values.size),
                            "max_absolute_preclip": float(np.max(np.abs(values))) if values.size else None})
    return results, valid, allowed


def prepare(output):
    if output.exists():
        load_protocol(output)
        print("existing sealed protocol verified", flush=True)
        return
    parent = base.read(previous.DEFAULT / "protocol.json")
    records = parent["records"]
    assert Counter(r["kind"] for r in records) == {"event": 633, "background": 12}
    assert len({r["event_id"] for r in records if r["kind"] == "event"}) == 65
    for record in records:
        base.verify(record["clean_input"])
    model = base.artifact(parent["models"]["cspp"]["path"])
    assert model["sha256"] == parent["models"]["cspp"]["sha256"]
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "role": "Paired post-hoc diagnostic on previously evaluated TexNet event and NCEDC background inputs; not independent parameter selection",
        "cases": list(LIMITS), "limits": [6, 12, 24, "positive infinity"], "baseline": "clip12",
        "models": parent["models"], "runtime": parent["runtime"], "cspp_thresholds": parent["cspp_thresholds"],
        "records": records, "change": "Only normalize_packet_ewma_causal clip argument; no weights, thresholds, quality-gate, scale, feature or candidate changes",
        "feature_caveat": "Existing absolute-value >=11.9 summary feature remains fixed, including under clip6/24/unlimited",
        "evaluation": {"primary_tolerance_seconds": 1.5, "secondary_tolerances_seconds": [0.5, 0.25],
                       "event_records": 633, "events": 65, "background_station_hours": 12,
                       "bootstrap": {"unit": "event", "draws": 10000, "seed": 20260909},
                       "matching": "At most one closest reference-P match per event record; all remaining candidates FP_reference, not manually confirmed seismic false alarms",
                       "clipping_scopes": ["All quality-valid samples including warmup", "Reference P <= sample time < P+1s", "P-1s <= sample time < P"],
                       "background": "Same fixed 12 station-hours in each case, not 48 independent station-hours; candidate burden not validated natural false alarm rate"},
        "no_training_no_threshold_tuning_no_production_changes": True,
        "parent_artifacts": [base.artifact(previous.DEFAULT / "protocol.json"), base.artifact(base.DEFAULT / "cspp_complete.json")],
    }
    output.mkdir(parents=True, exist_ok=False)
    base.write(output / "protocol.json", protocol)
    sources = {Path(__file__), Path(base.__file__), Path(previous.__file__),
               Path(__file__).with_name("aggregate_normalization_clip.m"), ROOT / "tests/test_normalization_clip_sensitivity.py",
               ROOT / "src/cspp_v27_reused_core_score_cache.py", ROOT / "src/cspp_v27_hybrid_score_cache.py"}
    sources.update(Path(s["path"]) for s in base.read(base.DEFAULT / "seal.json")["code"])
    base.write(output / "seal.json", {"protocol": base.artifact(output / "protocol.json"), "model": model,
                                      "sources": [base.artifact(p) for p in sorted(sources)],
                                      "inputs": [r["clean_input"] for r in records]})
    print("645 inputs and four clip limits sealed before inference", flush=True)


def load_protocol(output):
    sealed = base.read(output / "seal.json")
    for spec in [sealed["protocol"], sealed["model"], *sealed["sources"]]:
        base.verify(spec)
    protocol = base.read(output / "protocol.json")
    for spec in protocol["parent_artifacts"]:
        base.verify(spec)
    return protocol, sealed


def pilot(output):
    protocol, _ = load_protocol(output)
    ensemble = base.new_cspp(protocol)
    selected = protocol["records"][:2]
    cpu, comparisons = {}, {}
    for record in selected:
        wave = load_wave(record)
        for case in LIMITS:
            cpu[record["id"], case] = score(wave, record, protocol, ensemble, case, backend="cpu")
        original = core.score_waveform_packets(wave, start_epoch=record["start_epoch"], ensemble=ensemble, runtime=protocol["runtime"])
        comparisons[record["id"]+":unmodified_cpu"] = base.compare_cache(original, cpu[record["id"], "clip12"])
        assert all(value == 0 for value in comparisons[record["id"]+":unmodified_cpu"].values())
    hybrid.prepare_hybrid_ensemble(ensemble)
    for record in selected:
        wave = load_wave(record)
        for case in LIMITS:
            current = score(wave, record, protocol, ensemble, case)
            comparisons[record["id"]+":"+case] = base.compare_cache(cpu[record["id"], case], current)
            previous.compare_candidates(base.cspp_candidates(cpu[record["id"], case], record, protocol), base.cspp_candidates(current, record, protocol))
            changed = wave.copy()
            changed[:, 12000:] = 12345
            future = score(changed, record, protocol, ensemble, case)
            for name, values in current.arrays().items():
                assert np.allclose(values[:240], future.arrays()[name][:240], atol=1e-6, rtol=0, equal_nan=True)
        print(f"pilot {record['id']}: CPU/hybrid equivalence, clip12 parity and future perturbation passed", flush=True)
    base.write(output / "pilot.json", {"passed": True, "seal_sha256": base.sha256_file(output / "seal.json"), "comparisons": comparisons})


def run(output):
    protocol, _ = load_protocol(output)
    sealed_hash = base.sha256_file(output / "seal.json")
    pilot_result = base.read(output / "pilot.json")
    assert pilot_result["passed"] and pilot_result["seal_sha256"] == sealed_hash
    torch.mps.set_per_process_memory_fraction(0.3)
    ensemble = base.new_cspp(protocol)
    hybrid.prepare_hybrid_ensemble(ensemble)
    (output / "caches").mkdir(exist_ok=True)
    for index, record in enumerate(protocol["records"]):
        destination = output / "runs" / (record["id"]+".json")
        if destination.exists():
            done = base.read(destination)
            assert done["seal_sha256"] == sealed_hash and done["input_sha256"] == record["clean_input"]["sha256"]
            base.verify(done["cache"])
            continue
        started = time.monotonic()
        wave = load_wave(record)
        clipping, valid, allowed = clipping_audit(wave, record, protocol["runtime"])
        results, arrays = [], {}
        for case in LIMITS:
            cache = score(wave, record, protocol, ensemble, case)
            assert np.array_equal(cache.quality_valid, valid) and np.array_equal(cache.candidate_allowed, allowed)
            arrays.update({case+"__"+key: value for key, value in cache.arrays().items()})
            candidates = base.cspp_candidates(cache, record, protocol)
            results.append({"case": case, "candidates": candidates,
                            "allowed_seconds": float(allowed.sum()*0.5), "invalid_seconds": float((~valid).sum()*0.5)})
        historical_path = base.DEFAULT / "runs/cspp" / (record["id"]+".json")
        historical = base.read(historical_path)["results"][0]["candidates"]
        pick_difference = previous.compare_candidates(historical, results[1]["candidates"])
        cache_path = output / "caches" / (record["id"]+".npz")
        np.savez_compressed(cache_path, **arrays)
        base.write(destination, {"record_id": record["record_id"], "seal_sha256": sealed_hash,
                                 "input_sha256": record["clean_input"]["sha256"], "results": results, "clipping": clipping,
                                 "historical": base.artifact(historical_path), "clip12_max_pick_difference_s": pick_difference,
                                 "cache": base.artifact(cache_path), "wall_seconds": time.monotonic()-started})
        if index % 20 == 0 or index >= 633:
            print(f"clip comparison {index+1}/645 {time.monotonic()-started:.2f}s", flush=True)
        gc.collect()
        torch.mps.empty_cache()
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8*2**30
        assert torch.mps.driver_allocated_memory() < 4*2**30
    base.write(output / "complete.json", {"records": 645, "conditions_per_record": 4, "seal_sha256": sealed_hash,
                                          "outputs": [base.artifact(p) for p in sorted((output / "runs").glob("*.json"))]})


def export(output):
    protocol, _ = load_protocol(output)
    complete = base.read(output / "complete.json")
    assert complete["records"] == len(complete["outputs"]) == 645
    for spec in complete["outputs"]:
        base.verify(spec)
    candidates, health, clipping = [], [], []
    for record in protocol["records"]:
        data = base.read(output / "runs" / (record["id"]+".json"))
        base.verify(data["cache"])
        identity = {key: record[key] for key in ("id", "kind", "event_id", "station_key", "reference_p_epoch")}
        for result in data["results"]:
            fields = {**identity, "case": result["case"]}
            candidates.extend({**fields, "available_epoch": c["available_epoch"], "estimated_p_epoch": c["estimated_p_epoch"]} for c in result["candidates"])
            health.append({**fields, "candidate_count": len(result["candidates"]), "duration_seconds": record["sample_count"]/100,
                           "allowed_seconds": result["allowed_seconds"], "invalid_seconds": result["invalid_seconds"]})
        clipping.extend({**identity, **row} for row in data["clipping"])
    for filename, rows in [("candidates.csv", candidates), ("record_health.csv", health), ("clipping_counts.csv", clipping)]:
        base.write_csv(output / filename, rows)
    base.write(output / "export_complete.json", {"record_conditions": len(health), "clipping_rows": len(clipping),
                                                "files": [base.artifact(output / p) for p in ("candidates.csv", "record_health.csv", "clipping_counts.csv")]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "run", "export"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
