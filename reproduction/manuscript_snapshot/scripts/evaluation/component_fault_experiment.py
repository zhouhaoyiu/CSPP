#!/usr/bin/env python3
"""Paired, frozen-weight single-component fault diagnostic; no training."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import h5py
import numpy as np
from obspy import UTCDateTime
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/evaluation")]
from cspp.evaluation.score_cache import load_ensemble, score_waveform_packets, sha256_file
from cspp.evaluation.public_windows import (
    extract_closed_candidates, infer_first_available_annotation, load_public_model,
)
from cspp.streaming.candidate import PacketCandidateState
from cspp_v27_hybrid_score_cache import prepare_hybrid_ensemble, score_waveform_packets_hybrid
import build_v27_ncedc_continuous_score_cache as source_io
from eval_v27_ncedc_causal_public_baselines import orientation
from oriented_gappy_strong_motion_stream import stream_to_oriented_gappy_zne_array
from resilient_mseed import read_resilient_mseed

WORK = Path("workspace")
TX = WORK / "tx_external_event_validation_20260824"
DEFAULT = ROOT / "outputs/cspp_component_missing_20260908"
KEY = "C0_seed20261101"
CASES = ["clean", "zero_Z", "zero_N", "zero_E"]
MODELS = ["cspp", "phasenet_instance", "eqtransformer_original"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify(spec):
    actual = artifact(spec["path"])
    assert actual["bytes"] == int(spec["bytes"]) and actual["sha256"] == spec["sha256"], spec["path"]
    return Path(spec["path"])


def station(row):
    return ".".join(row[k] for k in ("network", "station", "location"))


def prepare(output):
    if (output / "protocol.json").exists():
        protocol = read(output / "protocol.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        original = read(TX / "protocol.json")
        public = read(TX / "causal_public_baseline_protocol.json")
        cohort_path = TX / "metadata/manifests/causal_public_baseline_cohort.csv"
        request_path = TX / "metadata/manifests/event_station_requests.csv"
        orient_path = TX / "metadata/manifests/event_station_orientations.csv"
        acq_path = TX / "metadata/manifests/causal_public_baseline_waveforms.csv"
        requests = {(r["event_id"], station(r)): r for r in rows(request_path)}
        orientations = source_io.orientation_groups(rows(orient_path))
        acquisitions = {r["event_id"]: r for r in rows(acq_path)}
        cohort = rows(cohort_path)
        assert len(cohort) == 633
        records = []
        for c in cohort:
            request = requests[c["event_id"], c["station_key"]]
            start = float(UTCDateTime(request["request_start_utc"]))
            reference = float(UTCDateTime(request["primary_pick_time_utc"]))
            fault_start = int(np.floor(((reference - start) - 0.5) / 0.5)) * 50
            records.append({"id": f"r{len(records):04d}", "kind": "event",
                            "record_id": f"{c['event_id']}:{c['station_key']}",
                            "event_id": c["event_id"], "station_key": c["station_key"],
                            "request": request, "start_epoch": start, "reference_p_epoch": reference,
                            "sample_count": 24000, "fault_starts": [fault_start],
                            "orientation": orientations[f"{c['event_id']}.{c['station_key']}"],
                            "acquisition": acquisitions[c["event_id"]]})
        bg_path = WORK / "ncedc_model_blind_20260817/continuous_scoring_hybrid_amendment.json"
        bg = read(bg_path)
        for spec in bg["inputs"].values():
            verify(spec)
        bg_requests = sorted(rows(bg["inputs"]["station_day_requests"]["path"]),
                             key=lambda r: (r["date_utc"], int(r["availability_rank"])))
        bg_acq = {source_io.station_day_id(r): r for r in rows(bg["inputs"]["acquisition_status"]["path"])}
        bg_orient = source_io.orientation_groups(rows(bg["inputs"]["station_day_orientations"]["path"]))
        selected = set()
        for request in bg_requests:
            identity = source_io.station_day_id(request)
            acquisition = bg_acq[identity]
            if station(request) in selected or acquisition["status"] == "download_or_parse_error":
                continue
            selected.add(station(request))
            request = dict(request, request_start_utc=request["date_utc"] + "T12:00:00Z",
                           request_end_utc=request["date_utc"] + "T13:00:00Z")
            records.append({"id": f"r{len(records):04d}", "kind": "background",
                            "record_id": identity + "__12-13UTC", "event_id": "",
                            "station_key": station(request), "request": request,
                            "start_epoch": float(UTCDateTime(request["request_start_utc"])),
                            "reference_p_epoch": None, "sample_count": 360000,
                            "fault_starts": list(range(6000, 360000 - 1000, 12000)),
                            "orientation": bg_orient[identity], "acquisition": acquisition})
        assert len(selected) == 12 and len(records) == 645
        inputs = [cohort_path, request_path, orient_path, acq_path, bg_path,
                  TX / "protocol.json", TX / "causal_public_baseline_protocol.json"]
        protocol = {
            "created_utc": datetime.now(UTC).isoformat(),
            "purpose": "controlled paired missing-component diagnostic on previously evaluated records; not a new blind test",
            "models": {"cspp": original["models"][KEY], **public["models"]},
            "runtime": original["runtime"], "cspp_thresholds": {"candidate": 0.1, "event": 0.975},
            "cases": CASES, "records": records, "inputs": [artifact(p) for p in inputs],
            "fault": {"duration_samples": 1000,
                      "event_alignment": "packet-aligned start 0.5 to <1.0 s before catalog P",
                      "zero": "replace only selected component with exact zeros",
                      "sample_rate_hz": 100,
                      "background": "first available day per station, fixed 12-13 UTC, starts 60+120k s; no score-based selection"},
            "evaluation": {"match_tolerance_seconds": 1.5, "one_true_pick_per_record": True,
                           "additional_candidates": "all count as false candidates against the single catalog reference; may include unlabeled arrivals",
                           "public_adapter": "native historical windows, unchanged blinding/stride, closed episodes; no CSPP quality or warmup gate; nonfinite input set to zero",
                           "latency": "first available matched candidate minus catalog P; adapter-specific, not optimal achievable latency",
                           "background_endpoint": "candidate burden and paired increments, not natural false-alarm rate",
                           "denominator": "all 633 event records per condition, including gated or failed records"},
            "limits": {"torch_threads": 3, "rss_gib": 8, "mps_driver_gib": 4},
            "no_training_no_threshold_search_no_manuscript_edits": True,
        }
        write(output / "protocol.json", protocol)
    for spec in protocol["inputs"]:
        verify(spec)
    source_stream = None
    source_path = None
    verified = set()
    for index, record in enumerate(protocol["records"]):
        destination = output / "inputs" / (record["id"] + "_clean.h5")
        if destination.exists():
            continue
        a = record["acquisition"]
        path = Path(a["mseed_file"])
        if path not in verified:
            verify({"path": str(path), "bytes": int(a["mseed_bytes"]), "sha256": a["mseed_sha256"]})
            verified.add(path)
        if source_path != path:
            source_stream = read_resilient_mseed(path).stream
            source_path = path
        r = record["request"]
        wave, _ = stream_to_oriented_gappy_zne_array(
            source_stream, {"network": r["network"], "station": r["station"], "location": r["location"],
                            "latitude": float(r["station_latitude"]), "longitude": float(r["station_longitude"])},
            orientation(record["orientation"]), UTCDateTime(r["request_start_utc"]),
            UTCDateTime(r["request_end_utc"]), target_sample_rate_hz=100, samples_per_packet=50)
        assert wave.shape == (3, record["sample_count"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.h5")
        with h5py.File(temporary, "w") as h:
            h.create_dataset("wave", data=np.asarray(wave, dtype=np.float32))
        temporary.replace(destination)
        if index % 25 == 0 or index >= 633:
            print(f"prepare {index + 1}/645 {record['record_id']}", flush=True)
    write(output / "prepared.json", {"records": len(protocol["records"]), "protocol_sha256": sha256_file(output / "protocol.json")})


def input_path(output, record):
    return output / "inputs" / (record["id"] + "_faults.h5")


def load_waves(output, record):
    with h5py.File(input_path(output, record), "r") as h:
        values = h["waves"][:]
    assert values.shape == (len(CASES), 3, record["sample_count"]), values.shape
    return values


def validate_faults(clean, values, record):
    assert np.array_equal(clean, values[0], equal_nan=True)
    mask = np.zeros(clean.shape[1], dtype=bool)
    for first in record["fault_starts"]:
        assert first >= 1000 and first + 1000 <= clean.shape[1]
        mask[first:first + 1000] = True
    for case in range(1, len(CASES)):
        axis = (case - 1) % 3
        other = [i for i in range(3) if i != axis]
        assert np.array_equal(values[case, other], clean[other], equal_nan=True)
        assert np.array_equal(values[case, axis, ~mask], clean[axis, ~mask], equal_nan=True)
        assert np.isfinite(values[case, axis, mask]).all()
        if case <= 3:
            assert np.count_nonzero(values[case, axis, mask]) == 0


def seal(output):
    assert not (output / "runs").exists(), "Cannot reseal after inference"
    protocol = read(output / "protocol.json")
    files = []
    for record in protocol["records"]:
        with h5py.File(output / "inputs" / (record["id"] + "_clean.h5")) as h:
            clean = h["wave"][:]
        validate_faults(clean, load_waves(output, record), record)
        files.append(artifact(input_path(output, record)))
    model_files = []
    for name, spec in protocol["models"].items():
        if name == "cspp":
            model_files.append(artifact(spec["path"]))
            assert model_files[-1]["sha256"] == spec["sha256"]
        else:
            for a in spec["artifacts"].values():
                model_files.append(artifact(verify(a)))
    code = [Path(__file__), Path(__file__).with_name("inject_component_faults.m")]
    code += list((ROOT / "cspp").rglob("*.py")) + list((ROOT / "src").glob("*.py"))
    write(output / "seal.json", {"sealed_utc": datetime.now(UTC).isoformat(),
                                "protocol": artifact(output / "protocol.json"),
                                "models": model_files, "code": [artifact(p) for p in code], "inputs": files,
                                "invariants": "clean exact; other two components exact; outside fault exact; zero exact; injected values finite"})
    print("sealed 645 paired inputs before model outputs", flush=True)


def load_seal(output):
    sealed = read(output / "seal.json")
    for item in [sealed["protocol"], *sealed["models"], *sealed["code"]]:
        verify(item)
    return read(output / "protocol.json"), sealed


def cspp_candidates(cache, record, protocol):
    state = PacketCandidateState(history_packets=4)
    candidates = []
    for i in range(len(cache.quality_valid)):
        candidate = state.step(
            packet_index=i, valid=bool(cache.quality_valid[i]), candidate_allowed=bool(cache.candidate_allowed[i]),
            candidate_probability=float(cache.candidate_probability[i]), event_probability=float(cache.event_probability[KEY][i]),
            arrival_age_fraction=float(cache.arrival_age_fraction[i]),
            candidate_threshold=protocol["cspp_thresholds"]["candidate"], event_threshold=protocol["cspp_thresholds"]["event"],
            start_epoch=record["start_epoch"], packet_seconds=0.5, max_arrival_age_packets=4)
        if candidate:
            candidates.append(candidate)
    return candidates


def health(cache, record, case):
    allowed = cache.candidate_allowed
    fault_mask = np.zeros(len(allowed), dtype=bool)
    recoveries = []
    for first in record["fault_starts"]:
        fault_mask[first // 50:(first + 1000) // 50] = True
        end = (first + 1000) // 50
        recovered = np.flatnonzero(allowed[end:])
        recoveries.append(None if not len(recovered) else float((recovered[0] + 1) * 0.5))
    return {"candidate_allowed_seconds": float(allowed.sum() * 0.5),
            "quality_invalid_seconds": float((~cache.quality_valid).sum() * 0.5),
            "disallowed_during_fault_seconds": float((~allowed & fault_mask).sum() * 0.5),
            "recovery_after_fault_seconds": recoveries, "diagnostics": cache.diagnostics}


def new_cspp(protocol):
    return load_ensemble({KEY: protocol["models"]["cspp"]}, device_name="cpu", torch_threads=3)


def compare_cache(a, b):
    aa = a.arrays()
    bb = b.arrays()
    result = {}
    for key, x in aa.items():
        y = bb[key]
        assert np.array_equal(np.isfinite(x), np.isfinite(y)), key
        finite = np.isfinite(x)
        if x.dtype == np.bool_:
            assert np.array_equal(x, y), key
            delta = 0.0
        else:
            delta = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
            assert delta <= 1e-4, (key, delta)
        result[key] = delta
    return result


def pilot(output):
    protocol, sealed = load_seal(output)
    selected = protocol["records"][:2]
    ensemble = new_cspp(protocol)
    cpu = {}
    for record in selected:
        verify(next(s for s in sealed["inputs"] if s["path"] == str(input_path(output, record))))
        for case, wave in enumerate(load_waves(output, record)):
            cpu[record["id"], case] = score_waveform_packets(wave, start_epoch=record["start_epoch"], ensemble=ensemble, runtime=protocol["runtime"])
    prepare_hybrid_ensemble(ensemble)
    comparisons = {}
    for record in selected:
        waves = load_waves(output, record)
        for case, wave in enumerate(waves):
            other = score_waveform_packets_hybrid(wave, start_epoch=record["start_epoch"], ensemble=ensemble, runtime=protocol["runtime"])
            original = cpu[record["id"], case]
            comparisons[f"{record['id']}:{CASES[case]}"] = compare_cache(original, other)
            left, right = cspp_candidates(original, record, protocol), cspp_candidates(other, record, protocol)
            assert [c["episode_start_packet"] for c in left] == [c["episode_start_packet"] for c in right]
            for a, b in zip(left, right, strict=True):
                assert abs(a["estimated_p_epoch"] - b["estimated_p_epoch"]) < 1e-4
            # Identical historical packets must produce identical causal outputs.
            cut = 12000
            changed = wave.copy()
            changed[:, cut:] = 12345
            future = score_waveform_packets_hybrid(changed, start_epoch=record["start_epoch"], ensemble=ensemble, runtime=protocol["runtime"])
            for name, array in other.arrays().items():
                assert np.allclose(array[:cut // 50], future.arrays()[name][:cut // 50], atol=1e-6, rtol=0, equal_nan=True), name
        print(f"pilot CSPP {record['id']} CPU/hybrid and future-independence passed", flush=True)
    del cpu, ensemble
    gc.collect()
    torch.mps.empty_cache()
    public_checks = {}
    for name in MODELS[1:]:
        spec = protocol["models"][name]
        model = load_public_model(spec, device=torch.device("mps"))
        wave = np.nan_to_num(load_waves(output, selected[0])[1], nan=0, posinf=0, neginf=0)
        cut = 12000
        changed = wave.copy()
        changed[:, cut:] = 12345
        annotations = [infer_first_available_annotation(model, spec, w, np.ones(w.shape[1] // 50, bool), device=torch.device("mps")) for w in (wave, changed)]
        mask = annotations[0].published & (annotations[0].available_packet < cut // 50)
        assert np.array_equal(annotations[0].available_packet[mask], annotations[1].available_packet[mask])
        assert np.allclose(annotations[0].p_probability[mask], annotations[1].p_probability[mask], atol=1e-6, rtol=0)
        assert annotations[0].diagnostics["executed_valid_windows"] > 0
        public_checks[name] = {"historical_samples_checked": int(mask.sum()), "zero_component_windows_executed": annotations[0].diagnostics["executed_valid_windows"]}
        del model, annotations
        gc.collect()
        torch.mps.empty_cache()
    write(output / "pilot.json", {"passed": True, "seal_sha256": sha256_file(output / "seal.json"),
                                 "cspp_cpu_hybrid_max_differences": comparisons, "public_future_independence": public_checks})


def run(output, name):
    protocol, sealed = load_seal(output)
    check = read(output / "pilot.json")
    assert check["passed"] and check["seal_sha256"] == sha256_file(output / "seal.json")
    torch.set_num_threads(3)
    torch.mps.set_per_process_memory_fraction(0.3)
    spec = protocol["models"][name]
    model = new_cspp(protocol) if name == "cspp" else load_public_model(spec, device=torch.device("mps"))
    if name == "cspp":
        prepare_hybrid_ensemble(model)
    for index, record in enumerate(protocol["records"]):
        destination = output / "runs" / name / (record["id"] + ".json")
        input_spec = next(s for s in sealed["inputs"] if s["path"] == str(input_path(output, record)))
        if destination.exists():
            prior = read(destination)
            assert prior["seal_sha256"] == sha256_file(output / "seal.json") and prior["input_sha256"] == input_spec["sha256"]
            continue
        verify(input_spec)
        started = time.monotonic()
        results = []
        for case, wave in enumerate(load_waves(output, record)):
            begin = time.monotonic()
            if name == "cspp":
                cache = score_waveform_packets_hybrid(wave, start_epoch=record["start_epoch"], ensemble=model, runtime=protocol["runtime"])
                candidates = cspp_candidates(cache, record, protocol)
                details = health(cache, record, case)
            else:
                finite_wave = np.nan_to_num(wave, nan=0, posinf=0, neginf=0)
                annotation = infer_first_available_annotation(model, spec, finite_wave, np.ones(wave.shape[1] // 50, bool), device=torch.device("mps"))
                candidates = extract_closed_candidates(annotation, method=spec["operating_points"][0], start_epoch=record["start_epoch"])
                details = {"diagnostics": annotation.diagnostics, "nonfinite_samples_sanitized": int((~np.isfinite(wave)).sum())}
            results.append({"case": CASES[case], "candidates": candidates, "health": details,
                            "wall_seconds": time.monotonic() - begin})
        write(destination, {"record_id": record["record_id"], "model": name, "seal_sha256": sha256_file(output / "seal.json"),
                            "input_sha256": input_spec["sha256"], "results": results,
                            "completed_utc": datetime.now(UTC).isoformat()})
        if index % 10 == 0 or index >= 633:
            print(f"{name} {index + 1}/645 {record['record_id']} {time.monotonic() - started:.2f}s", flush=True)
        gc.collect()
        torch.mps.empty_cache()
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8 * 2**30, "RSS limit"
        assert torch.mps.driver_allocated_memory() < 4 * 2**30, "MPS memory limit"
    write(output / f"{name}_complete.json", {"records": 645, "cases_per_record": len(CASES), "seal_sha256": sha256_file(output / "seal.json"),
                                           "outputs": [artifact(p) for p in sorted((output / "runs" / name).glob("*.json"))],
                                           "python": platform.python_version(), "torch": torch.__version__})


def export(output):
    protocol, _ = load_seal(output)
    candidates, health_rows = [], []
    for name in MODELS:
        completion = read(output / f"{name}_complete.json")
        assert completion["records"] == 645 and len(completion["outputs"]) == 645
        for file in completion["outputs"]:
            verify(file)
        for record in protocol["records"]:
            data = read(output / "runs" / name / (record["id"] + ".json"))
            assert [r["case"] for r in data["results"]] == CASES
            for result in data["results"]:
                base = {"id": record["id"], "kind": record["kind"], "event_id": record["event_id"],
                        "station_key": record["station_key"], "model": name, "case": result["case"],
                        "start_epoch": record["start_epoch"], "reference_p_epoch": record["reference_p_epoch"]}
                for c in result["candidates"]:
                    candidates.append({**base, "estimated_p_epoch": c["estimated_p_epoch"], "available_epoch": c["available_epoch"]})
                h = result["health"]
                recovery = h.get("recovery_after_fault_seconds", [])
                health_rows.append({**base, "candidate_count": len(result["candidates"]),
                                    "duration_seconds": record["sample_count"] / 100,
                                    "fault_count": len(record["fault_starts"]),
                                    "allowed_seconds": h.get("candidate_allowed_seconds"),
                                    "invalid_seconds": h.get("quality_invalid_seconds"),
                                    "disallowed_during_fault_seconds": h.get("disallowed_during_fault_seconds"),
                                    "recovery_mean_seconds": np.mean(recovery).item() if recovery and all(x is not None for x in recovery) else None,
                                    "unrecovered_faults": sum(x is None for x in recovery) if name == "cspp" else None,
                                    "published_samples": h["diagnostics"].get("published_samples"),
                                    "wall_seconds": result["wall_seconds"]})
    write_csv(output / "candidates.csv", candidates)
    write_csv(output / "record_health.csv", health_rows)
    write(output / "export_complete.json", {"health_rows": len(health_rows), "candidates": len(candidates),
                                            "files": [artifact(output / f) for f in ("candidates.csv", "record_health.csv")]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "seal", "pilot", "run", "export"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "run":
        assert args.model is not None
        run(args.output.resolve(), args.model)
    else:
        globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
