#!/usr/bin/env python3
"""Evaluate CSPP or public pickers on frozen K-NET/InstanceGM records."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import h5py
import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.filter import bandpass
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from v20c_causal_score_cache import (  # noqa: E402
    active_runs,
    load_frozen_models,
    release_device_cache,
    score_waveform_packets,
    sha256,
)
from v20c_event_snr import local_p_excess_power_snr  # noqa: E402
from v20c_public_window_baselines import (  # noqa: E402
    load_public_model,
)
from v21_causal_score_cache import emit_v21_candidates  # noqa: E402
from v21_score_shape_calibrator import (  # noqa: E402
    LinearScoreShapeCalibrator,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "experiments/v24_station_level_external_test_20260728/"
    "station_level_test_protocol.json"
)
MATCH_TOLERANCE_SECONDS = 1.5
ERROR_THRESHOLDS_SECONDS = (0.5, 1.0, 1.5)
OUTPUT_FIELDS = [
    "dataset",
    "model_key",
    "method_id",
    "comparison_mode",
    "availability_comparable",
    "source_index",
    "trace_name",
    "event_id",
    "station_id",
    "magnitude",
    "epicentral_distance_km",
    "p_arrival_sample",
    "sample_rate_hz",
    "n_samples",
    "waveform_sha256",
    "local_p_excess_power_snr_db",
    "candidate_count",
    "matched_candidate_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "within_0p5_seconds",
    "within_1p0_seconds",
    "within_1p5_seconds",
    "selected_estimated_p_seconds",
    "selected_available_seconds",
    "pick_error_seconds",
    "absolute_pick_error_seconds",
    "availability_latency_seconds",
    "technical_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--dataset", choices=("knet", "instancegm"), required=True
    )
    parser.add_argument(
        "--model",
        choices=("cspp", "phasenet_instance", "eqtransformer_original"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--limit", type=int, default=0, help="Smoke-test prefix only"
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_artifact(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
        raise RuntimeError(f"Artifact identity differs: {path}")
    expected_mtime = spec.get("mtime_ns")
    if expected_mtime is not None and path.stat().st_mtime_ns != int(
        expected_mtime
    ):
        raise RuntimeError(f"Artifact mtime differs: {path}")
    expected = spec.get("sha256")
    if expected and sha256(path) != str(expected):
        raise RuntimeError(f"Artifact SHA-256 differs: {path}")
    return path


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != (
        "domain_robust_v24_station_level_external_test_v1"
    ):
        raise RuntimeError("Unexpected V24 station-level test protocol")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256(source) != str(expected):
            raise RuntimeError(f"V24 evaluation code drifted: {relative}")
    for spec in protocol["inputs"].values():
        verify_artifact(spec)
    for spec in protocol["model_artifacts"].values():
        verify_artifact(spec)
    for model in protocol["public_models"].values():
        for spec in model["artifacts"].values():
            verify_artifact(spec)
    if float(protocol["fixed_test_metrics"]["match_tolerance_seconds"]) != (
        MATCH_TOLERANCE_SECONDS
    ):
        raise RuntimeError("Fixed-test match tolerance differs")
    return protocol, sha256(path)


def align_wave(wave: np.ndarray) -> np.ndarray:
    values = np.asarray(wave, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError(f"Expected at least three components, got {values.shape}")
    values = values[:3]
    sample_count = values.shape[1] - values.shape[1] % 50
    if sample_count < 50:
        raise ValueError("Trace contains no complete 0.5-s packet")
    return np.ascontiguousarray(values[:, :sample_count])


class WaveReader:
    def __init__(
        self, dataset: str, protocol: dict[str, Any]
    ) -> None:
        self.dataset = dataset
        self.handle = None
        if dataset == "knet":
            path = verify_artifact(protocol["source_stores"]["knet_waveforms"])
            self.handle = h5py.File(path, "r")
        else:
            path = verify_artifact(
                protocol["source_stores"]["instancegm_waveforms"]
            )
            self.handle = h5py.File(path, "r")

    @staticmethod
    def parse_bucket_location(trace_name: str) -> tuple[str, tuple[Any, ...]]:
        if "$" not in trace_name:
            return trace_name, (slice(None),)
        block, encoded = trace_name.split("$", maxsplit=1)
        location = []
        for token in encoded.split(","):
            if ":" not in token:
                location.append(int(token))
                continue
            values = [
                int(value) if value else None
                for value in token.split(":")
            ]
            location.append(slice(*values))
        return block, tuple(location)

    def read(self, row: dict[str, str]) -> np.ndarray:
        if self.dataset == "knet":
            return align_wave(
                self.handle["data"][row["trace_name"]][:].astype(np.float32)
            )
        block, location = self.parse_bucket_location(row["trace_name"])
        return align_wave(
            np.asarray(
                self.handle["data"][block][location],
                dtype=np.float32,
            )
        )

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def waveform_digest(wave: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(wave.dtype).encode("ascii"))
    digest.update(np.asarray(wave.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(wave).tobytes())
    return digest.hexdigest()


def load_cspp(protocol: dict[str, Any], device: str):
    backbone_spec = protocol["model_artifacts"]["backbone"]
    backbone, artifact_head, stored = load_frozen_models(
        backbone_spec["path"],
        protocol["model_artifacts"]["artifact_head"]["path"],
        device_name=device,
        torch_threads=int(protocol["execution"]["torch_threads"]),
        expected_backbone_sha256=backbone_spec["sha256"],
    )
    calibrator = LinearScoreShapeCalibrator.load(
        protocol["model_artifacts"]["score_shape_calibrator"]["path"]
    )
    return backbone, artifact_head, stored, calibrator


def score_cspp(
    wave: np.ndarray,
    protocol: dict[str, Any],
    loaded: tuple[Any, ...],
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    backbone, artifact_head, stored, calibrator = loaded
    cache = score_waveform_packets(
        wave,
        start_epoch=0.0,
        backbone=backbone,
        artifact_head=artifact_head,
        artifact_head_input_dim=int(stored["config"]["input_dim"]),
        runtime=protocol["runtime_contract"],
        device_name=device,
    )
    _, system_candidates = emit_v21_candidates(
        cache,
        start_epoch=0.0,
        runtime=protocol["runtime_contract"],
        calibrator=calibrator,
    )
    runtime = protocol["runtime_contract"]
    accepted = (
        np.asarray(cache.quality_valid, dtype=bool)
        & np.isfinite(cache.p_probability)
        & np.isfinite(cache.event_probability)
        & (
            cache.p_probability
            >= float(runtime["candidate_threshold"])
        )
        & (
            cache.event_probability
            >= float(runtime["event_threshold"])
        )
    )
    starts, ends = active_runs(accepted)
    packet_seconds = float(runtime["packet_duration_seconds"])
    cooldown_packets = int(
        math.ceil(
            float(runtime["emission_cooldown_seconds"])
            / packet_seconds
        )
    )
    age_scale_seconds = (
        float(runtime["max_arrival_age_packets"]) * packet_seconds
    )
    core_candidates = []
    last_emitted = -(10**12)
    for episode, (first, last) in enumerate(
        zip(starts, ends, strict=True), start=1
    ):
        packet = int(first)
        if packet - last_emitted < cooldown_packets:
            continue
        available = (packet + 1) * packet_seconds
        arrival_age = float(cache.arrival_age_fraction[packet])
        core_candidates.append(
            {
                "episode_index": episode,
                "episode_start_packet": packet,
                "episode_end_packet": int(last),
                "packet_index": packet,
                "available_epoch": available,
                "estimated_p_epoch": (
                    available - arrival_age * age_scale_seconds
                ),
                "p_probability": float(cache.p_probability[packet]),
                "event_probability": float(
                    cache.event_probability[packet]
                ),
            }
        )
        last_emitted = packet
    return {
        "cspp_backbone_fixed": core_candidates,
        "cspp_system_cold_start": system_candidates,
    }


def score_public(
    wave: np.ndarray,
    protocol: dict[str, Any],
    model_key: str,
    model: torch.nn.Module,
) -> dict[str, list[dict[str, Any]]]:
    """Run the released full-record annotation interface.

    These fixed event excerpts do not contain the 30/60 s prehistory needed
    for a cold-start causal replay of PhaseNet/EQTransformer. Their standard
    full-record picks are therefore used only for endpoint accuracy. Causal
    availability is compared separately on GeoNet windows with real prehistory.
    """
    spec = protocol["public_models"][model_key]
    values = np.asarray(wave, dtype=np.float32)
    filter_spec = spec.get("trailing_window_filter")
    if filter_spec is not None:
        values = bandpass(
            values,
            freqmin=float(filter_spec["freqmin_hz"]),
            freqmax=float(filter_spec["freqmax_hz"]),
            df=100.0,
            corners=int(filter_spec["corners"]),
            zerophase=bool(filter_spec["zerophase"]),
            axis=-1,
        ).astype(np.float32, copy=False)
    stream = Stream()
    for component, channel in zip(
        values, ("HNZ", "HNN", "HNE"), strict=True
    ):
        trace = Trace(np.ascontiguousarray(component))
        trace.stats.starttime = UTCDateTime(0)
        trace.stats.sampling_rate = 100.0
        trace.stats.network = "XX"
        trace.stats.station = "V24"
        trace.stats.location = ""
        trace.stats.channel = channel
        stream += trace
    annotations = model.annotate(
        stream,
        copy=True,
        batch_size=int(spec["execution_batch_size"]),
        overlap=int(spec["native_overlap_samples"]),
        blinding=tuple(int(value) for value in spec["blinding_samples"]),
        strict=True,
        flexible_horizontal_components=False,
    )
    p_annotations = annotations.select(
        channel=f"{spec['class']}_P"
    )
    if len(p_annotations) != 1:
        raise RuntimeError(
            f"Expected one public P annotation, got {len(p_annotations)}"
        )
    detection_annotations = annotations.select(
        channel=f"{spec['class']}_Detection"
    )
    if len(detection_annotations) > 1:
        raise RuntimeError("Expected at most one detection annotation")

    by_method: dict[str, list[dict[str, Any]]] = {}
    for method in spec["operating_points"]:
        picks = model.picks_from_annotations(
            p_annotations,
            float(method["p_threshold"]),
            "P",
        )
        candidates = []
        for pick in picks:
            estimated = float(pick.peak_time)
            detection_probability = None
            if detection_annotations:
                detection_trace = detection_annotations[0]
                sample = int(
                    round(
                        (
                            pick.peak_time
                            - detection_trace.stats.starttime
                        )
                        * detection_trace.stats.sampling_rate
                    )
                )
                if 0 <= sample < len(detection_trace.data):
                    detection_probability = float(
                        detection_trace.data[sample]
                    )
            detection_threshold = method.get("detection_threshold")
            if detection_threshold is not None and (
                detection_probability is None
                or detection_probability < float(detection_threshold)
            ):
                continue
            candidates.append(
                {
                    "estimated_p_epoch": estimated,
                    "available_epoch": math.nan,
                    "p_probability": float(pick.peak_value),
                    "detection_probability": detection_probability,
                }
            )
        by_method[str(method["id"])] = candidates
    return by_method


def scored_row(
    source: dict[str, str],
    *,
    dataset: str,
    model_key: str,
    method_id: str,
    comparison_mode: str,
    availability_comparable: bool,
    wave: np.ndarray,
    candidates: list[dict[str, Any]],
    technical_error: str = "",
) -> dict[str, Any]:
    sample_rate = float(source["sample_rate_hz"])
    reference = float(source["p_arrival_sample"]) / sample_rate
    candidate_errors = [
        float(candidate["estimated_p_epoch"]) - reference
        for candidate in candidates
    ]
    matched_indices = [
        index
        for index, error in enumerate(candidate_errors)
        if abs(error) <= MATCH_TOLERANCE_SECONDS
    ]
    selected_index = (
        min(
            matched_indices,
            key=lambda index: (
                abs(candidate_errors[index]),
                float(candidates[index]["estimated_p_epoch"]),
                index,
            ),
        )
        if matched_indices
        else None
    )
    hit = selected_index is not None
    selected = candidates[selected_index] if hit else None
    error = candidate_errors[selected_index] if hit else math.nan
    snr = local_p_excess_power_snr(
        wave,
        p_sample=int(round(float(source["p_arrival_sample"]))),
        sample_rate_hz=sample_rate,
        window_seconds=1.0,
    )
    snr_db = snr["snr_db"] if snr["valid"] else None
    return {
        "dataset": dataset,
        "model_key": model_key,
        "method_id": method_id,
        "comparison_mode": comparison_mode,
        "availability_comparable": int(availability_comparable),
        "source_index": int(source["source_index"]),
        "trace_name": source["trace_name"],
        "event_id": source["event_id"],
        "station_id": source["station_id"],
        "magnitude": float(source["magnitude"]),
        "epicentral_distance_km": float(
            source["epicentral_distance_km"]
        ),
        "p_arrival_sample": float(source["p_arrival_sample"]),
        "sample_rate_hz": sample_rate,
        "n_samples": int(wave.shape[1]),
        "waveform_sha256": waveform_digest(wave),
        "local_p_excess_power_snr_db": (
            math.nan if snr_db is None else float(snr_db)
        ),
        "candidate_count": len(candidates),
        "matched_candidate_count": len(matched_indices),
        "true_positive": int(hit),
        "false_positive": len(candidates) - int(hit),
        "false_negative": int(not hit),
        "within_0p5_seconds": int(hit and abs(error) <= 0.5),
        "within_1p0_seconds": int(hit and abs(error) <= 1.0),
        "within_1p5_seconds": int(hit),
        "selected_estimated_p_seconds": (
            math.nan if selected is None else selected["estimated_p_epoch"]
        ),
        "selected_available_seconds": (
            math.nan
            if selected is None or not availability_comparable
            else selected["available_epoch"]
        ),
        "pick_error_seconds": error,
        "absolute_pick_error_seconds": (
            math.nan if not hit else abs(error)
        ),
        "availability_latency_seconds": (
            math.nan
            if selected is None or not availability_comparable
            else float(selected["available_epoch"]) - reference
        ),
        "technical_error": technical_error,
    }


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    protocol, protocol_hash = load_protocol(args.protocol.resolve())
    manifest_spec = protocol["fixed_test_records"][args.dataset]
    records = read_csv(verify_artifact(manifest_spec))
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if args.limit:
        records = records[: args.limit]
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else Path(protocol["execution"]["fixed_test_output_root"]).resolve()
    )
    run_root = output_root / args.dataset / args.model
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)

    reader = WaveReader(args.dataset, protocol)
    device = torch.device(args.device)
    if args.model == "cspp":
        loaded = load_cspp(protocol, args.device)
        public_model = None
    else:
        loaded = None
        public_model = load_public_model(
            protocol["public_models"][args.model],
            device=device,
        )
    output_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for serial, source in enumerate(records, start=1):
            wave: np.ndarray | None = None
            try:
                wave = reader.read(source)
                if float(source["p_arrival_sample"]) >= wave.shape[1]:
                    raise ValueError("Reference P lies outside aligned waveform")
                if args.model == "cspp":
                    methods = score_cspp(
                        wave, protocol, loaded, args.device
                    )
                else:
                    methods = score_public(
                        wave,
                        protocol,
                        args.model,
                        public_model,
                    )
                for method_id, candidates in methods.items():
                    comparison_mode = (
                        "causal_backbone_packet_replay"
                        if method_id == "cspp_backbone_fixed"
                        else (
                            "causal_full_system_cold_start"
                            if method_id == "cspp_system_cold_start"
                            else "standard_full_record_annotation"
                        )
                    )
                    output_rows.append(
                        scored_row(
                            source,
                            dataset=args.dataset,
                            model_key=args.model,
                            method_id=method_id,
                            comparison_mode=comparison_mode,
                            availability_comparable=(
                                args.model == "cspp"
                            ),
                            wave=wave,
                            candidates=candidates,
                        )
                    )
            except Exception as error:  # preserve technical failures as misses
                if wave is None:
                    wave = np.zeros((3, 50), dtype=np.float32)
                method_ids = (
                    [
                        "cspp_backbone_fixed",
                        "cspp_system_cold_start",
                    ]
                    if args.model == "cspp"
                    else [
                        str(method["id"])
                        for method in protocol["public_models"][
                            args.model
                        ]["operating_points"]
                    ]
                )
                for method_id in method_ids:
                    comparison_mode = (
                        "causal_backbone_packet_replay"
                        if method_id == "cspp_backbone_fixed"
                        else (
                            "causal_full_system_cold_start"
                            if method_id == "cspp_system_cold_start"
                            else "standard_full_record_annotation"
                        )
                    )
                    output_rows.append(
                        scored_row(
                            source,
                            dataset=args.dataset,
                            model_key=args.model,
                            method_id=method_id,
                            comparison_mode=comparison_mode,
                            availability_comparable=(
                                args.model == "cspp"
                            ),
                            wave=wave,
                            candidates=[],
                            technical_error=(
                                f"{type(error).__name__}: {error}"
                            ),
                        )
                    )
            if serial % 25 == 0 or serial == len(records):
                print(
                    f"{args.dataset} {args.model} "
                    f"[{serial}/{len(records)}]",
                    flush=True,
                )
                if args.device == "mps":
                    release_device_cache(args.device)
                gc.collect()
    finally:
        reader.close()

    result_path = run_root / "records.csv"
    write_csv_atomic(result_path, output_rows)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "dataset": args.dataset,
        "model_key": args.model,
        "records_requested": len(records),
        "output_rows": len(output_rows),
        "technical_failures": sum(
            bool(row["technical_error"]) for row in output_rows
        ),
        "methods": sorted({row["method_id"] for row in output_rows}),
        "comparison_mode": (
            [
                "causal_backbone_packet_replay",
                "causal_full_system_cold_start",
            ]
            if args.model == "cspp"
            else ["standard_full_record_annotation"]
        ),
        "availability_comparable": args.model == "cspp",
        "device": args.device,
        "limit": args.limit,
        "runtime_seconds": time.time() - started,
        "worker_max_resident_set_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "records_artifact": {
            "path": str(result_path.resolve()),
            "bytes": result_path.stat().st_size,
            "sha256": sha256(result_path),
        },
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
