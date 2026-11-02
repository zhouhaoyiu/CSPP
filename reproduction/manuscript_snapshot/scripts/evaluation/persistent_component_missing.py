#!/usr/bin/env python3
"""CSPP record-long component absence, initialized with missing input."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).parent)]
import component_fault_experiment as base
import component_gate_ablation as gate

DEFAULT = ROOT / "outputs/cspp_persistent_component_missing_20260908"
VARIANTS = ("original_gate", "two_component_gate")


def input_path(output, record):
    return output / "inputs" / (record["id"] + ".h5")


def load_waves(output, record):
    with h5py.File(input_path(output, record)) as handle:
        values = handle["waves"][:]
    assert values.shape == (4, 3, record["sample_count"])
    return values


def same_bits(a, b):
    return np.array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))


def validate_inputs(clean, values):
    assert clean.dtype == values.dtype == np.float32
    assert values.shape == (4, *clean.shape)
    assert same_bits(clean, values[0]), "clean control changed"
    for axis in range(3):
        masked = values[axis+1]
        assert np.isfinite(masked[axis]).all() and np.count_nonzero(masked[axis]) == 0, "absence must span the entire record"
        retained = [i for i in range(3) if i != axis]
        assert same_bits(masked[retained], clean[retained]), "retained channels changed"


def prepare(output):
    if (output / "protocol.json").exists():
        print("protocol already exists; use seal to verify prepared input", flush=True)
        return
    _, parent, parent_seal = gate.load_protocol(gate.DEFAULT)
    assert base.read(gate.DEFAULT / "verification.json")["passed"]
    specs = {Path(s["path"]).name: s for s in parent_seal["inputs"]}
    records = []
    for index, record in enumerate(parent["records"]):
        paired = specs[base.input_path(base.DEFAULT, record).name]
        base.verify(paired)
        clean_path = base.DEFAULT / "inputs" / (record["id"] + "_clean.h5")
        with h5py.File(clean_path) as handle:
            clean = handle["wave"][:]
        with h5py.File(paired["path"]) as handle:
            assert same_bits(clean, handle["waves"][0])
        kept = {key: record[key] for key in (
            "id", "record_id", "kind", "event_id", "station_key", "start_epoch", "reference_p_epoch", "sample_count")}
        kept["clean_input"] = base.artifact(clean_path)
        records.append(kept)
        if index % 100 == 0 or index == 644:
            print(f"source control verified {index+1}/645", flush=True)
    runtime = dict(parent["runtime"])
    runtime["normalization"] = "Per-packet component demeaning and preceding shared-channel EWMA scale, initialized independently from each condition's first packet"
    protocol = {
        "created_utc": datetime.now(UTC).isoformat(),
        "role": "Corrected persistent-component diagnostic on previously evaluated data; not a new blind cohort or trained missing-component model",
        "parent": str(base.DEFAULT), "variants": list(VARIANTS), "cases": base.CASES,
        "models": {"cspp": parent["models"]["cspp"]}, "runtime": runtime,
        "cspp_thresholds": parent["cspp_thresholds"], "records": records,
        "missingness": {
            "scope": "Selected component is exact zero from sample zero to the last sample, including all initialization and warmup",
            "constructor_inputs": "Clean waveform, channel index, and length only; no reference arrival or onset/offset selection",
            "retained_channels": "Bitwise unchanged, including preexisting nonfinite values",
            "no_recovery": True, "no_clean_prefix": True,
            "meaning": "Zero-filled absent Z/N/E model input after orientation; not a physical zero-motion observation or all instrument fault types",
        },
        "state": "Independent normalization and GRU/candidate initialization at each record-condition; original initial 20-packet warmup remains",
        "evaluation": {
            "primary_tolerance_seconds": 1.5, "secondary_tolerances_seconds": [0.5, 0.25],
            "records_per_event_condition": 633, "events": 65, "background_station_hours": 12,
            "matching": "At most one reference-P hit per record; all other candidates unmatched to this reference, not necessarily false seismic arrivals",
            "bootstrap": {"unit": "event", "paired_against": "same variant clean", "draws": 10000, "seed": 20260910},
            "background": "Candidate burden on fixed 12 station-hours; not a verified natural false-alarm rate",
        },
        "prior_artifacts": [base.artifact(base.DEFAULT / "protocol.json"), base.artifact(base.DEFAULT / "seal.json"),
                            base.artifact(base.DEFAULT / "cspp_complete.json"), base.artifact(gate.DEFAULT / "protocol.json")],
        "no_training_no_threshold_tuning_no_production_changes": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    base.write(output / "protocol.json", protocol)
    print("persistent absence protocol saved before input construction and inference", flush=True)


def seal(output):
    assert not (output / "runs").exists()
    protocol = base.read(output / "protocol.json")
    files = []
    for record in protocol["records"]:
        base.verify(record["clean_input"])
        with h5py.File(record["clean_input"]["path"]) as handle:
            clean = handle["wave"][:]
        validate_inputs(clean, load_waves(output, record))
        files.append(base.artifact(input_path(output, record)))
    sources = [Path(__file__), Path(__file__).with_name("inject_persistent_component_missing.m"),
               ROOT / "tests/test_persistent_component_missing.py", Path(gate.__file__), Path(base.__file__)]
    parent_seal = base.read(base.DEFAULT / "seal.json")
    sources += [Path(s["path"]) for s in parent_seal["code"]]
    sources = sorted(set(sources))
    model = base.artifact(protocol["models"]["cspp"]["path"])
    assert model["sha256"] == protocol["models"]["cspp"]["sha256"]
    base.write(output / "seal.json", {"protocol": base.artifact(output / "protocol.json"), "model": model,
                                      "sources": [base.artifact(s) for s in sources], "inputs": files,
                                      "validated_records": len(files), "no_P_timing_in_constructor": True})
    print("sealed 645 full-length four-condition inputs", flush=True)


def load_protocol(output):
    seal = base.read(output / "seal.json")
    for item in [seal["protocol"], seal["model"], *seal["sources"]]:
        base.verify(item)
    protocol = base.read(output / "protocol.json")
    for item in protocol["prior_artifacts"]:
        base.verify(item)
    return protocol, seal


def compare_candidates(left, right):
    assert len(left) == len(right)
    difference = 0.0
    for a, b in zip(left, right, strict=True):
        assert a["episode_start_packet"] == b["episode_start_packet"]
        assert a["available_epoch"] == b["available_epoch"]
        difference = max(difference, abs(a["estimated_p_epoch"]-b["estimated_p_epoch"]))
    assert difference < 1e-4
    return difference


def pilot(output):
    protocol, _ = load_protocol(output)
    selected = protocol["records"][:2]
    ensemble = base.new_cspp(protocol)
    cpu = {}
    for record in selected:
        for case, wave in enumerate(load_waves(output, record)):
            for variant in VARIANTS:
                cpu[record["id"], case, variant] = gate.score(
                    wave, record, protocol, ensemble, backend="cpu", ablation=variant == VARIANTS[1])
    base.prepare_hybrid_ensemble(ensemble)
    comparisons = {}
    for record in selected:
        for case, wave in enumerate(load_waves(output, record)):
            for variant in VARIANTS:
                ablation = variant == VARIANTS[1]
                cache = gate.score(wave, record, protocol, ensemble, ablation=ablation)
                prior = cpu[record["id"], case, variant]
                comparisons[f"{record['id']}:{case}:{variant}"] = base.compare_cache(prior, cache)
                compare_candidates(base.cspp_candidates(prior, record, protocol), base.cspp_candidates(cache, record, protocol))
                changed = wave.copy()
                retained = [axis for axis in range(3) if case == 0 or axis != case-1]
                changed[retained, 12000:] = 12345
                future = gate.score(changed, record, protocol, ensemble, ablation=ablation)
                for name, values in cache.arrays().items():
                    assert np.allclose(values[:240], future.arrays()[name][:240], atol=1e-6, rtol=0, equal_nan=True)
                if case and not ablation:
                    assert not cache.quality_valid.any() and not cache.candidate_allowed.any()
            if case == 0:
                a = cpu[record["id"], case, VARIANTS[0]]
                b = cpu[record["id"], case, VARIANTS[1]]
                base.compare_cache(a, b)
        print(f"pilot {record['id']}: backend parity, original rejection, clean equivalence and causality passed", flush=True)
    base.write(output / "pilot.json", {"passed": True, "comparisons": comparisons,
                                       "seal_sha256": base.sha256_file(output / "seal.json")})


def run(output):
    protocol, sealed = load_protocol(output)
    seal_hash = base.sha256_file(output / "seal.json")
    pilot_result = base.read(output / "pilot.json")
    assert pilot_result["passed"] and pilot_result["seal_sha256"] == seal_hash
    specs = {Path(s["path"]).name: s for s in sealed["inputs"]}
    torch.mps.set_per_process_memory_fraction(0.3)
    ensemble = base.new_cspp(protocol)
    base.prepare_hybrid_ensemble(ensemble)
    for index, record in enumerate(protocol["records"]):
        destination = output / "runs" / (record["id"] + ".json")
        spec = specs[input_path(output, record).name]
        if destination.exists():
            prior = base.read(destination)
            assert prior["seal_sha256"] == seal_hash and prior["input_sha256"] == spec["sha256"]
            continue
        base.verify(spec)
        started = time.monotonic()
        results = []
        for case, wave in enumerate(load_waves(output, record)):
            for variant in VARIANTS:
                cache = gate.score(wave, record, protocol, ensemble, ablation=variant == VARIANTS[1])
                candidates = base.cspp_candidates(cache, record, protocol)
                allowed = np.flatnonzero(cache.candidate_allowed)
                if case and variant == VARIANTS[0]:
                    assert not cache.quality_valid.any() and not candidates
                results.append({
                    "case": base.CASES[case], "variant": variant, "candidates": candidates,
                    "health": {"allowed_seconds": float(cache.candidate_allowed.sum()*0.5),
                               "invalid_seconds": float((~cache.quality_valid).sum()*0.5),
                               "first_allowed_seconds": float((allowed[0]+1)*0.5) if len(allowed) else None,
                               "accepted_zero_packets": cache.diagnostics["quality_flags"].get("one_zero_component_accepted", 0),
                               "diagnostics": cache.diagnostics},
                })
        historical = base.read(base.DEFAULT / "runs/cspp" / (record["id"] + ".json"))["results"][0]["candidates"]
        parity = [compare_candidates(historical, r["candidates"]) for r in results if r["case"] == "clean"]
        base.write(destination, {"record_id": record["record_id"], "seal_sha256": seal_hash,
                                 "input_sha256": spec["sha256"], "results": results,
                                 "clean_max_pick_difference_s": max(parity), "wall_seconds": time.monotonic()-started})
        if index % 20 == 0 or index >= 633:
            print(f"persistent CSPP {index+1}/645 {time.monotonic()-started:.2f}s", flush=True)
        gc.collect()
        torch.mps.empty_cache()
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8*2**30
        assert torch.mps.driver_allocated_memory() < 4*2**30
    base.write(output / "complete.json", {"records": 645, "conditions_per_record": 8, "seal_sha256": seal_hash,
                                          "outputs": [base.artifact(p) for p in sorted((output / "runs").glob("*.json"))]})


def export(output):
    protocol, _ = load_protocol(output)
    complete = base.read(output / "complete.json")
    assert complete["records"] == len(complete["outputs"]) == 645
    for item in complete["outputs"]:
        base.verify(item)
    candidates, health = [], []
    for record in protocol["records"]:
        data = base.read(output / "runs" / (record["id"] + ".json"))
        assert len(data["results"]) == 8
        for result in data["results"]:
            identity = {key: record[key] for key in ("id", "kind", "event_id", "station_key", "reference_p_epoch")}
            identity.update(variant=result["variant"], case=result["case"])
            candidates.extend({**identity, "available_epoch": c["available_epoch"], "estimated_p_epoch": c["estimated_p_epoch"]} for c in result["candidates"])
            details = result["health"]
            health.append({**identity, "candidate_count": len(result["candidates"]), "duration_seconds": record["sample_count"]/100,
                           **{key: details[key] for key in ("allowed_seconds", "invalid_seconds", "first_allowed_seconds", "accepted_zero_packets")}})
    base.write_csv(output / "candidates.csv", candidates)
    base.write_csv(output / "record_health.csv", health)
    base.write(output / "export_complete.json", {"record_conditions": len(health),
                                                "files": [base.artifact(output / p) for p in ("candidates.csv", "record_health.csv")]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "seal", "pilot", "run", "export"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
