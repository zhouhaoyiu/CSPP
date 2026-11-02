#!/usr/bin/env python3
"""Prepare, audit and score supplied BU100 event records with frozen CSPP."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts/evaluation")]
from causal_fir_decimator import FIR_COEFFICIENTS, causal_decimate_200_to_100
from cspp.evaluation.score_cache import tensor_state_digest
from scipy.signal import firwin, lfilter
from component_fault_experiment import (
    new_cspp, cspp_candidates, compare_cache, score_waveform_packets,
    prepare_hybrid_ensemble, score_waveform_packets_hybrid,
)

OUT = ROOT / "outputs/cspp_bu100_20260910"
SOURCE = Path("external_data/chinese_restricted")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    return {"path": str(path), "sha256": sha(path), "bytes": Path(path).stat().st_size}


def write_csv(path, records):
    if not records:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("Frozen protocol already exists; refusing to overwrite")
    labels = sorted((SOURCE / "震例").rglob("*P波*.txt"))
    recent = SOURCE / "到目前所有包括震例数据/2022"
    for folder in sorted(recent.iterdir()):
        if folder.name.startswith(("20220601", "20220610", "20220905")):
            labels.extend(sorted(folder.rglob("*P波*.txt")))
    labels = [p for p in labels if not p.name.startswith("._")]
    records, issues, inventory, sources = [], [], [], {}
    for label in labels:
        intensity = "烈度计" in label.name
        scope = label.parent
        if intensity and (scope / "烈度计").is_dir():
            scope = scope / "烈度计"
        cohort = label.parent.name
        if cohort in ("强震", "烈度计"):
            cohort = label.parent.parent.name
        elif cohort in ("5.8", "6.0"):
            cohort = label.parent.parent.parent.name + "_" + cohort
        files = [p for p in scope.rglob("*") if p.is_file() and not p.name.startswith("._")]
        grouped = defaultdict(lambda: defaultdict(list))
        for p in files:
            if intensity and p.suffix.lower() == ".txt" and p != label and "P波" not in p.name:
                grouped[p.stem]["txt"].append(p)
            elif not intensity and p.suffix.upper() in (".UD", ".NS", ".EW"):
                grouped[p.stem][{".UD": "z", ".NS": "n", ".EW": "e"}[p.suffix.upper()]].append(p)
        pairs = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        assert all(len(x) == 2 for x in pairs), label
        assert len({x[0] for x in pairs}) == len(pairs), label
        sources[str(label)] = artifact(label)
        needed = ["txt"] if intensity else ["z", "n", "e"]
        matched = 0
        for key, seconds in pairs:
            paths = grouped.get(key, {})
            if any(len(paths.get(k, [])) != 1 for k in needed):
                issues.append({"cohort": cohort, "instrument": "intensity" if intensity else "strong_motion",
                               "record_key": key, "reason": "missing_or_ambiguous_component", "label_path": str(label)})
                continue
            matched += 1
            record = {"id": f"r{len(records):05d}", "cohort": cohort,
                      "instrument": "intensity" if intensity else "strong_motion",
                      "record_key": key, "reference_p_seconds": float(seconds),
                      "label_path": str(label), "path_z": "", "path_n": "", "path_e": "", "path_txt": ""}
            for k in needed:
                p = paths[k][0]
                record["path_" + k] = str(p)
                if str(p) not in sources:
                    sources[str(p)] = artifact(p)
            records.append(record)
        extra = sorted(set(grouped) - {x[0] for x in pairs})
        for key in extra:
            issues.append({"cohort": cohort, "instrument": "intensity" if intensity else "strong_motion",
                           "record_key": key, "reason": "waveform_without_reference", "label_path": str(label)})
        inventory.append({"cohort": cohort, "instrument": "intensity" if intensity else "strong_motion",
                          "label_rows": len(pairs), "waveform_keys": len(grouped), "matched": matched,
                          "unlabelled_waveforms": len(extra), "label_path": str(label)})
        print(f"paired {cohort} {'intensity' if intensity else 'strong'}: {matched}/{len(pairs)}", flush=True)
    base = read(ROOT / "outputs/cspp_random_component_missing_20260908/protocol.json")
    model = base["models"]["cspp"]
    model_provenance = {"original": dict(model), "route": "original_checkpoint"}
    if not Path(model["path"]).exists():
        release = ROOT / "releases/CSPP_0.1.0_20260907/CSPP_0.1.0_named"
        manifest_path = release / "models/model_manifest.json"
        export = next(s for s in read(manifest_path)["checkpoints"] if s["seed"] == model["seed"])
        assert export["source_sha256"] == model["sha256"] and export["tensors_bitwise_equal"]
        local = release / export["public_path"]
        assert sha(local) == export["public_sha256"]
        checkpoint = torch.load(local, map_location="cpu", weights_only=False)
        assert tensor_state_digest(checkpoint["model"]) == export["tensor_state_sha256"]
        model.update(artifact(local))
        model_provenance.update(route="tensor_identical_release_export", export=export,
                                manifest=artifact(manifest_path), original_live_comparison=False)
    assert sha(model["path"]) == model["sha256"]
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": base["models"],
                "cspp_thresholds": base["cspp_thresholds"], "runtime": base["runtime"],
                "records": records, "fir_coefficients": FIR_COEFFICIENTS.tolist(), "model_provenance": model_provenance,
                "design": {"target": "supplied-label event-record replay; no training or tuning",
                           "order": "ZNE", "units": "preserve source; intensity physical units unverified",
                           "resampling": "100 Hz unchanged; 200 Hz causal 31-tap FIR, phase 0, first 15 target samples NaN",
                           "fir_group_delay_seconds": 0.075, "delay_correction": "none for primary picks or availability",
                           "reference": "supplied seconds from record start; provenance and independent manual accuracy unverified",
                           "matching": "earliest emitted candidate with absolute P error <= tolerance; at most one match per record",
                           "tolerances_seconds": [1.5, 0.5, 0.25],
                           "packet_tail": "convert full trace; omit <50 final samples from inference without padding",
                           "limitations": ["selected event records, not continuous noise", "unmatched candidates are not confirmed false alarms",
                                           "no independent train/test waveform-overlap audit", "source good-data selection retained"]}}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "protocol.json", protocol)
    write(OUT / "source_manifest.json", list(sources.values()))
    write_csv(OUT / "inventory.csv", inventory)
    write_csv(OUT / "pairing_issues.csv", issues)
    write(OUT / "freeze.json", {"protocol": artifact(OUT / "protocol.json"),
                               "sources": artifact(OUT / "source_manifest.json"), "model": model})
    print(f"Frozen {len(records)} records, {len(issues)} pairing issues, {len(sources)} source files", flush=True)


def load_protocol():
    seal = read(OUT / "freeze.json")
    for spec in seal.values():
        assert sha(spec["path"]) == spec["sha256"], spec["path"]
    return read(OUT / "protocol.json")


def amend_resampling():
    protocol = load_protocol()
    assert not (OUT / "inference_code_freeze.json").exists(), "Cannot amend after inference starts"
    assert "resampling_500hz" not in protocol, "Already amended"
    write(OUT / "initial_parser_attempt/protocol_before_500hz_amendment.json", protocol)
    protocol["resampling_500hz"] = {
        "source_sample_rate_hz": 500, "target_sample_rate_hz": 100,
        "coefficients": firwin(77, 40, window="hamming", fs=500).tolist(),
        "num_taps": 77, "cutoff_hz": 40, "group_delay_seconds": 0.076,
        "decimation_phase": 0, "initial_target_samples_withheld": 16,
        "reason": "One native 500 Hz record discovered by header parsing; no model scores observed",
        "rule": "Same 40 Hz Hamming-window FIR family as 200 Hz; nearest longer odd length to 0.15 seconds support",
        "amended_utc": datetime.now(timezone.utc).isoformat()}
    protocol["design"]["resampling"] += "; 500 Hz causal 77-tap FIR at 40 Hz, phase 0, first 16 target samples NaN"
    write(OUT / "protocol.json", protocol)
    seal = read(OUT / "freeze.json")
    seal["protocol"] = artifact(OUT / "protocol.json")
    write(OUT / "freeze.json", seal)


def raw_wave(record):
    # Independent reader used for conversion QA, not for producing converted data.
    if record["instrument"] == "intensity":
        return np.loadtxt(record["path_txt"], delimiter=",", comments="#").T[[0, 2, 1]]
    components = []
    for c in "zne":
        with Path(record["path_" + c]).open(errors="replace") as f:
            header = "".join(next(f) for _ in range(16))
            n = int(re.search(r"NO\. OF POINTS:\s*(\d+)", header)[1])
            values = np.fromstring(f.read(), sep=" ")
            if values.size != n:
                assert n < values.size == ((n + 7) // 8) * 8 and np.all(values[n:] == 0)
            components.append(values[:n])
    return np.stack(components)


def audit():
    protocol = load_protocol()
    status = read(OUT / "conversion_status.json")
    lookup = {r["id"]: r for r in protocol["records"]}
    for i, spec in enumerate(read(OUT / "source_manifest.json")):
        assert sha(spec["path"]) == spec["sha256"], spec["path"]
        if i % 500 == 0:
            print(f"source hash check {i}", flush=True)
    checks, seen, duplicates = [], {}, []
    with h5py.File(OUT / "waveforms_100hz_zne.h5", "r") as h:
        for s in status:
            if s["status"] != "converted":
                continue
            wave = h[s["id"] + "/wave"][:]
            assert wave.shape == (3, s["target_samples"]), (s["id"], wave.shape)
            assert s["target_sample_rate_hz"] == 100
            assert 0 <= s["reference_p_seconds"] < wave.shape[1] / 100
            digest = hashlib.sha256(wave.tobytes()).hexdigest()
            if digest in seen:
                duplicates.append({"id": s["id"], "duplicate_of": seen[digest]})
            seen[digest] = s["id"]
            # One record per cohort/instrument/rate plus all first 12 records.
            key = (s["cohort"], s["instrument"], s["source_sample_rate_hz"])
            if key not in {tuple(c["group"]) for c in checks} or len(checks) < 12:
                raw = raw_wave(lookup[s["id"]])
                rate = s["source_sample_rate_hz"]
                if rate == 100:
                    expected = raw.astype(np.float32)
                elif rate == 200:
                    expected = np.stack([causal_decimate_200_to_100(c) for c in raw])
                elif rate == 500:
                    expected = lfilter(firwin(77, 40, fs=500), [1], raw, axis=1)[:, ::5].astype(np.float32)
                    expected[:, :16] = np.nan
                else:
                    raise AssertionError(rate)
                assert np.array_equal(np.isfinite(wave), np.isfinite(expected)), s["id"]
                finite = np.isfinite(wave)
                delta = float(np.max(np.abs(wave[finite] - expected[finite])))
                assert np.allclose(wave, expected, rtol=2e-6, atol=1e-6, equal_nan=True), (s["id"], delta)
                checks.append({"id": s["id"], "group": key, "max_abs_error": delta})
    assert not duplicates, duplicates
    write(OUT / "conversion_audit.json", {"source_hashes_verified": True, "roundtrip_checks": checks,
                                          "exact_waveform_duplicates": duplicates,
                                          "status_counts": dict(Counter(s["status"] for s in status)),
                                          "waveforms": artifact(OUT / "waveforms_100hz_zne.h5")})
    print("Conversion audit passed", flush=True)


def wave_for(h, record):
    wave = h[record["id"] + "/wave"][:]
    return wave[:, :wave.shape[1] // 50 * 50]


def infer():
    protocol = load_protocol()
    audit_result = read(OUT / "conversion_audit.json")
    spec = audit_result["waveforms"]
    assert sha(spec["path"]) == spec["sha256"]
    records = [s for s in read(OUT / "conversion_status.json") if s["status"] == "converted"]
    code_paths = [Path(__file__), ROOT / "scripts/evaluation/convert_bu100_matlab.m",
                  ROOT / "scripts/evaluation/summarize_bu100_matlab.m",
                  ROOT / "scripts/evaluation/component_fault_experiment.py",
                  ROOT / "cspp/streaming/candidate.py", ROOT / "cspp/evaluation/score_cache.py",
                  ROOT / "src/cspp_v27_reused_core_score_cache.py",
                  ROOT / "src/cspp_v27_hybrid_score_cache.py", ROOT / "src/causal_fir_decimator.py"]
    code_seal = OUT / "inference_code_freeze.json"
    if code_seal.exists():
        for item in read(code_seal):
            assert sha(item["path"]) == item["sha256"], item["path"]
    else:
        write(code_seal, [artifact(p) for p in code_paths])
    ensemble = new_cspp(protocol)
    pilot_records = [next(s for s in records if s["source_sample_rate_hz"] == rate)
                     for rate in sorted({r["source_sample_rate_hz"] for r in records})]
    with h5py.File(spec["path"], "r") as h:
        cpu = [score_waveform_packets(wave_for(h, s), start_epoch=0, ensemble=ensemble, runtime=protocol["runtime"])
               for s in pilot_records]
        prepare_hybrid_ensemble(ensemble)
        comparisons = []
        for s, original in zip(pilot_records, cpu, strict=True):
            wave = wave_for(h, s)
            other = score_waveform_packets_hybrid(wave, start_epoch=0, ensemble=ensemble, runtime=protocol["runtime"])
            comparisons.append({"id": s["id"], "max_differences": compare_cache(original, other)})
            a = cspp_candidates(original, {"start_epoch": 0}, protocol)
            b = cspp_candidates(other, {"start_epoch": 0}, protocol)
            assert [x["episode_start_packet"] for x in a] == [x["episode_start_packet"] for x in b]
            assert all(abs(x["estimated_p_epoch"] - y["estimated_p_epoch"]) < 1e-4 for x, y in zip(a, b, strict=True))
            cut = (wave.shape[1] // 100) * 50
            changed = wave.copy()
            changed[:, cut:] = 12345
            future = score_waveform_packets_hybrid(changed, start_epoch=0, ensemble=ensemble, runtime=protocol["runtime"])
            for name, values in other.arrays().items():
                assert np.allclose(values[:cut // 50], future.arrays()[name][:cut // 50], rtol=0, atol=1e-6, equal_nan=True)
        write(OUT / "inference_pilot.json", {"CPU_hybrid_checks": comparisons, "future_independence": True})
        (OUT / "record_results").mkdir(exist_ok=True)
        started = time.monotonic()
        for i, s in enumerate(records):
            dest = OUT / "record_results" / (s["id"] + ".json")
            if dest.exists():
                continue
            wave = wave_for(h, s)
            cache = score_waveform_packets_hybrid(wave, start_epoch=0, ensemble=ensemble, runtime=protocol["runtime"])
            candidates = cspp_candidates(cache, {"start_epoch": 0}, protocol)
            reference = s["reference_p_seconds"]
            j = min(int(reference / 0.5), len(cache.candidate_allowed) - 1)
            result = {"id": s["id"], "cohort": s["cohort"], "instrument": s["instrument"],
                      "record_key": s["record_key"], "reference_p_seconds": reference,
                      "source_sample_rate_hz": s["source_sample_rate_hz"],
                      "absolute_start_provided": s["absolute_start_provided"],
                      "evaluated_seconds": wave.shape[1] / 100, "tail_samples_omitted": s["target_samples"] - wave.shape[1],
                      "reference_in_evaluated_samples": reference < wave.shape[1] / 100,
                      "candidate_allowed_at_reference": bool(cache.candidate_allowed[j]),
                      "quality_valid_seconds": float(cache.quality_valid.sum() * 0.5),
                      "candidate_allowed_seconds": float(cache.candidate_allowed.sum() * 0.5),
                      "candidate_count": len(candidates), "candidates": candidates, "diagnostics": cache.diagnostics}
            write(dest, result)
            if (i + 1) % 25 == 0 or i == len(records) - 1:
                print(f"CSPP {i + 1}/{len(records)} elapsed={time.monotonic() - started:.1f}s", flush=True)
    flat = []
    for s in records:
        r = read(OUT / "record_results" / (s["id"] + ".json"))
        entry = {k: v for k, v in r.items() if k not in ("candidates", "diagnostics")}
        for tolerance, suffix in [(1.5, "1500ms"), (0.5, "500ms"), (0.25, "250ms")]:
            matches = [c for c in r["candidates"] if abs(c["estimated_p_epoch"] - r["reference_p_seconds"]) <= tolerance]
            entry["matched_" + suffix] = int(bool(matches))
            entry["error_" + suffix] = matches[0]["estimated_p_epoch"] - r["reference_p_seconds"] if matches else ""
            entry["latency_" + suffix] = matches[0]["available_epoch"] - r["reference_p_seconds"] if matches else ""
        flat.append(entry)
    write_csv(OUT / "record_metrics.csv", flat)
    code = [Path(__file__), ROOT / "scripts/evaluation/convert_bu100_matlab.m", ROOT / "scripts/evaluation/summarize_bu100_matlab.m",
            ROOT / "cspp/streaming/candidate.py", ROOT / "src/cspp_v27_reused_core_score_cache.py",
            ROOT / "src/cspp_v27_hybrid_score_cache.py", ROOT / "src/causal_fir_decimator.py"]
    write(OUT / "inference_complete.json", {"records": len(flat), "metrics": artifact(OUT / "record_metrics.csv"),
                                           "code": [artifact(p) for p in code]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "amend_resampling", "audit", "infer"])
    args = parser.parse_args()
    {"prepare": prepare, "amend_resampling": amend_resampling, "audit": audit, "infer": infer}[args.stage]()
