#!/usr/bin/env python3
"""Score one frozen V27C operating point on fixed retrospective records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluation.analyze_v27_geonet_temporal_selection import (  # noqa: E402
    extract_candidates,
    support_with_resets,
)
from cspp_v27_reused_core_score_cache import (  # noqa: E402
    V27ScoreCache,
    load_ensemble,
    score_waveform_packets,
)
from v20c_event_snr import (  # noqa: E402
    DEFINITION_ID,
    SNR_BIN_INVALID,
    local_p_excess_power_snr,
)
from scripts.evaluation import v27_runner_lease as runner_lease  # noqa: E402


ALLOWED_OUTPUT_ROOT = Path(
    "workspace"
).resolve()
FORBIDDEN_VOLUME = Path("external_volume")
DEFAULT_PROTOCOL = (
    ALLOWED_OUTPUT_ROOT
    / "protocol_freeze/cspp_v27c_fixed_retrospective_protocol_20260729.json"
)
PROTOCOL_ID = "cspp_v27c_fixed_retrospective_benchmark_v1"
MODEL_KEY = "cspp_v27c_primary"

OUTPUT_FIELDS = [
    "dataset",
    "model_key",
    "selected_arm",
    "primary_seed",
    "source_index",
    "trace_name",
    "event_id",
    "station_id",
    "magnitude",
    "epicentral_distance_km",
    "p_arrival_sample",
    "p_precontext_seconds",
    "p_precontext_stratum",
    "sample_rate_hz",
    "manifest_n_samples",
    "evaluated_n_samples",
    "waveform_sha256",
    "snr_definition_id",
    "snr_valid",
    "local_p_excess_power_snr_db",
    "snr_bin",
    "packet_count",
    "quality_valid_packets",
    "quality_invalid_packets",
    "candidate_allowed_packets",
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
    "technical_failure",
    "technical_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--expected-protocol-sha256",
        required=True,
        help="Required SHA-256 of the frozen retrospective protocol",
    )
    parser.add_argument(
        "--dataset", choices=("knet", "instancegm"), required=True
    )
    parser.add_argument("--runner-lease", type=Path, required=True)
    parser.add_argument("--attempt-output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def reject_forbidden_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if is_within(resolved, FORBIDDEN_VOLUME):
        raise RuntimeError(f"WD_BLACK is forbidden for this benchmark: {resolved}")
    return resolved


def require_output_path(path: str | Path, allowed_root: Path) -> Path:
    resolved = reject_forbidden_path(path)
    if not is_within(resolved, allowed_root):
        raise RuntimeError(
            f"Benchmark output must remain under {allowed_root}: {resolved}"
        )
    return resolved


def artifact(path: str | Path, *, hash_file: bool = True) -> dict[str, Any]:
    resolved = reject_forbidden_path(path)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(resolved) if hash_file else None,
    }


def verify_artifact(
    spec: Mapping[str, Any],
    *,
    require_sha256: bool,
    require_mtime: bool = False,
) -> Path:
    path = reject_forbidden_path(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if int(spec["bytes"]) != int(stat.st_size):
        raise RuntimeError(f"Artifact byte count differs: {path}")
    if require_mtime and int(spec["mtime_ns"]) != int(stat.st_mtime_ns):
        raise RuntimeError(f"Artifact mtime differs: {path}")
    expected = spec.get("sha256")
    if require_sha256 and not expected:
        raise RuntimeError(f"Artifact has no SHA-256: {path}")
    if expected and sha256_file(path) != str(expected):
        raise RuntimeError(f"Artifact SHA-256 differs: {path}")
    return path


def load_protocol(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    protocol_path = reject_forbidden_path(path)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    actual_hash = sha256_file(protocol_path)
    if actual_hash != expected_sha256.lower():
        raise RuntimeError("Retrospective protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected retrospective protocol")
    point = protocol.get("operating_point", {})
    if (
        int(point.get("count", 0)) != 1
        or bool(point.get("threshold_scan_permitted", True))
        or not math.isclose(
            float(point.get("candidate_threshold", math.nan)),
            0.1,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("Protocol does not contain one frozen operating point")
    if protocol.get("model", {}).get("ensemble_members") != 1:
        raise RuntimeError("Retrospective evaluator requires one checkpoint")
    if protocol.get("execution", {}).get("device") != "cpu":
        raise RuntimeError("Retrospective protocol must use CPU")

    allowed_root = Path(
        protocol["execution"]["allowed_output_root"]
    ).resolve()
    if allowed_root != ALLOWED_OUTPUT_ROOT:
        raise RuntimeError("Unexpected retrospective output root")
    require_output_path(
        protocol["execution"]["score_output_root"], allowed_root
    )
    require_output_path(
        protocol["execution"]["analysis_output_root"], allowed_root
    )

    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != str(expected):
            raise RuntimeError(f"Frozen evaluation code drifted: {relative}")
    for key in (
        "selection_freeze",
        "development_protocol",
        "v24_fixed_test_protocol",
    ):
        verify_artifact(
            protocol[key], require_sha256=True, require_mtime=False
        )
    selection_result = protocol.get("selection_result")
    if selection_result:
        verify_artifact(
            selection_result, require_sha256=True, require_mtime=False
        )
    verify_artifact(
        protocol["model"]["checkpoint"],
        require_sha256=True,
        require_mtime=False,
    )
    verify_artifact(
        protocol["model"]["training_selection_reference"],
        require_sha256=True,
        require_mtime=False,
    )
    for spec in protocol["fixed_test_records"].values():
        verify_artifact(spec, require_sha256=True, require_mtime=False)
    for spec in protocol["source_stores"].values():
        verify_artifact(
            spec, require_sha256=False, require_mtime=True
        )
    references = protocol.get("selected_waveform_hash_references", {})
    if set(references) != {"knet", "instancegm"}:
        raise RuntimeError("Waveform-hash references are incomplete")
    for spec in references.values():
        verify_artifact(
            spec, require_sha256=True, require_mtime=False
        )
    overlap = protocol.get("training_test_overlap_audit", {})
    overlap_datasets = overlap.get("datasets", {})
    if (
        overlap.get("status") != "zero_source_index_intersection"
        or set(overlap_datasets) != {"knet", "instancegm"}
    ):
        raise RuntimeError("Training/test overlap audit differs")
    for dataset in ("knet", "instancegm"):
        entries = overlap_datasets[dataset]
        if not entries:
            raise RuntimeError("Training/test overlap audit is empty")
        if any(
            int(entry.get("intersection", -1)) != 0
            for entry in entries.values()
        ):
            raise RuntimeError("Training/test overlap audit is nonzero")
    contracts = protocol.get("waveform_input_contracts", {})
    if set(contracts) != {"knet", "instancegm"}:
        raise RuntimeError("Waveform input contracts are incomplete")
    for dataset, contract in contracts.items():
        if (
            contract.get("component_order") != "ZNE"
            or contract.get("dimension_order") != "CW"
            or int(contract.get("component_count", 0)) != 3
        ):
            raise RuntimeError(
                f"Unexpected waveform input contract: {dataset}"
            )
    return protocol, actual_hash


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def align_wave(
    wave: np.ndarray,
    *,
    manifest_n_samples: int,
) -> np.ndarray:
    values = np.asarray(wave, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError(
            f"Expected exactly three components, got {values.shape}"
        )
    if values.shape[1] < manifest_n_samples:
        raise ValueError(
            f"Waveform is shorter than its manifest: {values.shape[1]} "
            f"< {manifest_n_samples}"
        )
    sample_count = int(manifest_n_samples)
    sample_count -= sample_count % 50
    if sample_count < 50:
        raise ValueError("Trace contains no complete 0.5-s packet")
    return np.ascontiguousarray(values[:, :sample_count])


class WaveReader:
    """Open exactly one frozen source store without preloading waveforms."""

    def __init__(self, dataset: str, protocol: Mapping[str, Any]) -> None:
        self.dataset = dataset
        self.input_contract = protocol["waveform_input_contracts"][dataset]
        key = "knet_waveforms" if dataset == "knet" else "instancegm_waveforms"
        path = verify_artifact(
            protocol["source_stores"][key],
            require_sha256=False,
            require_mtime=True,
        )
        self.handle = h5py.File(path, "r")
        if dataset == "instancegm":
            data_format = self.handle.get("data_format")
            if data_format is None:
                raise RuntimeError("InstanceGM source lacks data_format")
            component_order = self._hdf5_text(
                data_format.get("component_order")
            )
            dimension_order = self._hdf5_text(
                data_format.get("dimension_order")
            )
            if (
                component_order
                != self.input_contract["component_order"]
                or dimension_order
                != self.input_contract["dimension_order"]
            ):
                raise RuntimeError(
                    "InstanceGM source data format differs from protocol"
                )

    @staticmethod
    def _hdf5_text(dataset: h5py.Dataset | None) -> str:
        if dataset is None:
            raise RuntimeError("HDF5 text dataset is absent")
        value = dataset[()]
        if isinstance(value, bytes):
            return value.decode("ascii")
        if isinstance(value, str):
            return value
        raise RuntimeError(f"Expected HDF5 text scalar: {dataset.name}")

    @staticmethod
    def parse_bucket_location(
        trace_name: str,
    ) -> tuple[str, tuple[Any, ...]]:
        if "$" not in trace_name:
            return trace_name, (slice(None),)
        block, encoded = trace_name.split("$", maxsplit=1)
        location: list[Any] = []
        for token in encoded.split(","):
            if ":" not in token:
                location.append(int(token))
                continue
            values = [
                int(value) if value else None for value in token.split(":")
            ]
            location.append(slice(*values))
        return block, tuple(location)

    def read(self, row: Mapping[str, str]) -> np.ndarray:
        expected = int(float(row["n_samples"]))
        if self.dataset == "knet":
            if row.get("component_order") != "ZNE":
                raise ValueError("K-NET record component order is not ZNE")
            raw = self.handle["data"][row["trace_name"]][:]
        else:
            block, location = self.parse_bucket_location(row["trace_name"])
            raw = self.handle["data"][block][location]
        return align_wave(raw, manifest_n_samples=expected)

    def close(self) -> None:
        self.handle.close()


def waveform_digest(wave: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(wave, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def candidates_from_cache(
    cache: V27ScoreCache,
    *,
    event_head_key: str,
    operating_point: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> list[dict[str, float | int]]:
    support = support_with_resets(
        cache.candidate_probability,
        cache.candidate_allowed,
        cache.quality_valid,
        threshold=float(operating_point["candidate_threshold"]),
        history_packets=int(runtime["candidate_history_packets"]),
    )
    return extract_candidates(
        support=support,
        valid=cache.quality_valid,
        event_probability=cache.event_probability[event_head_key],
        arrival_age=cache.arrival_age_fraction,
        event_threshold=float(operating_point["event_threshold"]),
        start_epoch=0.0,
        packet_seconds=float(runtime["packet_duration_seconds"]),
        max_arrival_age_packets=float(
            runtime["max_arrival_age_packets"]
        ),
    )


def _source_value(
    source: Mapping[str, str],
    key: str,
    cast: Any,
    default: Any = "",
) -> Any:
    value = source.get(key, default)
    return cast(value) if value not in ("", None) else default


def record_result(
    source: Mapping[str, str],
    *,
    dataset: str,
    selected_arm: str,
    primary_seed: int,
    candidates: list[Mapping[str, float | int]],
    tolerance_seconds: float,
    wave: np.ndarray | None,
    diagnostics: Mapping[str, Any] | None,
    technical_error: str = "",
) -> dict[str, Any]:
    sample_rate = float(source["sample_rate_hz"])
    p_sample = float(source["p_arrival_sample"])
    reference_seconds = p_sample / sample_rate
    errors = [
        float(candidate["estimated_p_epoch"]) - reference_seconds
        for candidate in candidates
    ]
    matching = [
        index
        for index, error in enumerate(errors)
        if abs(error) <= tolerance_seconds
    ]
    selected_index = (
        min(
            matching,
            key=lambda index: (
                abs(errors[index]),
                float(candidates[index]["available_epoch"]),
                index,
            ),
        )
        if matching
        else None
    )
    hit = selected_index is not None and not technical_error
    selected = candidates[selected_index] if hit else None
    error = errors[selected_index] if hit else math.nan

    if wave is None:
        snr = {
            "definition_id": DEFINITION_ID,
            "valid": False,
            "snr_db": None,
            "snr_bin": SNR_BIN_INVALID,
        }
        digest = ""
        evaluated_n_samples = 0
    else:
        snr = local_p_excess_power_snr(
            wave,
            p_sample=int(round(p_sample)),
            sample_rate_hz=sample_rate,
            window_seconds=1.0,
        )
        digest = waveform_digest(wave)
        evaluated_n_samples = int(wave.shape[1])

    packet_count = (
        int(diagnostics.get("packet_count", 0)) if diagnostics else 0
    )
    valid_packets = (
        int(diagnostics.get("quality_valid_packets", 0))
        if diagnostics
        else 0
    )
    p_precontext = reference_seconds
    return {
        "dataset": dataset,
        "model_key": MODEL_KEY,
        "selected_arm": selected_arm,
        "primary_seed": int(primary_seed),
        "source_index": int(source["source_index"]),
        "trace_name": source["trace_name"],
        "event_id": source["event_id"],
        "station_id": source["station_id"],
        "magnitude": _source_value(source, "magnitude", float, math.nan),
        "epicentral_distance_km": _source_value(
            source, "epicentral_distance_km", float, math.nan
        ),
        "p_arrival_sample": p_sample,
        "p_precontext_seconds": p_precontext,
        "p_precontext_stratum": (
            "lt_10_seconds" if p_precontext < 10.0 else "ge_10_seconds"
        ),
        "sample_rate_hz": sample_rate,
        "manifest_n_samples": int(float(source["n_samples"])),
        "evaluated_n_samples": evaluated_n_samples,
        "waveform_sha256": digest,
        "snr_definition_id": str(snr["definition_id"]),
        "snr_valid": int(bool(snr["valid"])),
        "local_p_excess_power_snr_db": (
            math.nan if snr["snr_db"] is None else float(snr["snr_db"])
        ),
        "snr_bin": str(snr["snr_bin"]),
        "packet_count": packet_count,
        "quality_valid_packets": valid_packets,
        "quality_invalid_packets": packet_count - valid_packets,
        "candidate_allowed_packets": (
            int(diagnostics.get("candidate_allowed_packets", 0))
            if diagnostics
            else 0
        ),
        "candidate_count": len(candidates) if not technical_error else 0,
        "matched_candidate_count": len(matching) if not technical_error else 0,
        "true_positive": int(hit),
        "false_positive": (
            len(candidates) - int(hit) if not technical_error else 0
        ),
        "false_negative": int(not hit),
        "within_0p5_seconds": int(hit and abs(error) <= 0.5),
        "within_1p0_seconds": int(hit and abs(error) <= 1.0),
        "within_1p5_seconds": int(hit),
        "selected_estimated_p_seconds": (
            math.nan
            if selected is None
            else float(selected["estimated_p_epoch"])
        ),
        "selected_available_seconds": (
            math.nan
            if selected is None
            else float(selected["available_epoch"])
        ),
        "pick_error_seconds": error,
        "absolute_pick_error_seconds": (
            math.nan if not hit else abs(error)
        ),
        "availability_latency_seconds": (
            math.nan
            if selected is None
            else float(selected["available_epoch"]) - reference_seconds
        ),
        "technical_failure": int(bool(technical_error)),
        "technical_error": technical_error,
    }


def score_loaded_record(
    source: Mapping[str, str],
    wave: np.ndarray,
    *,
    dataset: str,
    ensemble: Any,
    protocol: Mapping[str, Any],
    event_head_key: str = "primary",
) -> dict[str, Any]:
    cache = score_waveform_packets(
        wave,
        start_epoch=0.0,
        ensemble=ensemble,
        runtime=protocol["runtime"],
    )
    candidates = candidates_from_cache(
        cache,
        event_head_key=event_head_key,
        operating_point=protocol["operating_point"],
        runtime=protocol["runtime"],
    )
    return record_result(
        source,
        dataset=dataset,
        selected_arm=protocol["model"]["selected_arm"],
        primary_seed=int(protocol["model"]["primary_seed"]),
        candidates=candidates,
        tolerance_seconds=float(
            protocol["metrics"]["match_tolerance_seconds"]
        ),
        wave=wave,
        diagnostics=cache.diagnostics,
    )


def write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
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


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    protocol, protocol_hash = load_protocol(
        args.protocol,
        expected_sha256=args.expected_protocol_sha256,
    )
    dataset = args.dataset
    manifest_path = verify_artifact(
        protocol["fixed_test_records"][dataset],
        require_sha256=True,
        require_mtime=False,
    )
    records = read_csv(manifest_path)
    expected = int(protocol["fixed_test_cohorts"][dataset]["records"])
    if len(records) != expected:
        raise RuntimeError(
            f"{dataset} manifest has {len(records)} rows, expected {expected}"
        )

    allowed_root = Path(
        protocol["execution"]["allowed_output_root"]
    ).resolve()
    output_root = require_output_path(
        protocol["execution"]["score_output_root"], allowed_root
    )
    run_root = require_output_path(output_root / dataset, allowed_root)
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    attempt_root = require_output_path(
        args.attempt_output_root, allowed_root
    )
    if (
        attempt_root.parent != run_root.parent
        or not attempt_root.name.startswith(f".{run_root.name}.attempt-")
        or not attempt_root.is_dir()
    ):
        raise RuntimeError("Runner attempt directory differs")
    runner_lease.verify_lease(
        args.runner_lease,
        protocol_sha256=protocol_hash,
        unit={"dataset": dataset},
        evaluator_script=Path(__file__).resolve(),
        attempt_root=attempt_root,
        published_root=run_root,
    )
    unexpected = [
        path.name
        for path in attempt_root.iterdir()
        if path.name != "runner_lease.json"
        and not path.name.startswith("._")
    ]
    if unexpected:
        raise RuntimeError(
            f"Runner attempt directory is not empty: {unexpected}"
        )

    checkpoint = protocol["model"]["checkpoint"]
    ensemble = load_ensemble(
        {
            "primary": {
                "path": checkpoint["path"],
                "sha256": checkpoint["sha256"],
                "verifier_feature_mode": checkpoint[
                    "verifier_feature_mode"
                ],
            }
        },
        device_name="cpu",
        torch_threads=int(protocol["execution"]["torch_threads"]),
    )
    if set(ensemble.event_heads) != {"primary"}:
        raise RuntimeError("Evaluator loaded more than the primary checkpoint")

    rows: list[dict[str, Any]] = []
    reader = WaveReader(dataset, protocol)
    started = time.time()
    try:
        for serial, source in enumerate(records, start=1):
            wave: np.ndarray | None = None
            try:
                wave = reader.read(source)
                if float(source["p_arrival_sample"]) >= wave.shape[1]:
                    raise ValueError(
                        "Reference P lies outside the evaluated waveform"
                    )
                row = score_loaded_record(
                    source,
                    wave,
                    dataset=dataset,
                    ensemble=ensemble,
                    protocol=protocol,
                )
            except Exception as error:
                row = record_result(
                    source,
                    dataset=dataset,
                    selected_arm=protocol["model"]["selected_arm"],
                    primary_seed=int(protocol["model"]["primary_seed"]),
                    candidates=[],
                    tolerance_seconds=float(
                        protocol["metrics"]["match_tolerance_seconds"]
                    ),
                    wave=wave,
                    diagnostics=None,
                    technical_error=f"{type(error).__name__}: {error}",
                )
            rows.append(row)
            if serial % 50 == 0 or serial == len(records):
                print(f"{dataset} [{serial}/{len(records)}]", flush=True)
    finally:
        reader.close()

    attempt_records_path = attempt_root / "records.csv"
    write_csv_once(attempt_records_path, rows)
    final_records_path = run_root / "records.csv"
    records_artifact = {
        **artifact(attempt_records_path),
        "path": str(final_records_path.resolve()),
    }
    summary = {
        "summary_id": "cspp_v27c_fixed_retrospective_score_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "dataset": dataset,
        "model_key": MODEL_KEY,
        "selected_arm": protocol["model"]["selected_arm"],
        "primary_seed": int(protocol["model"]["primary_seed"]),
        "operating_point": protocol["operating_point"],
        "checkpoint": protocol["model"]["checkpoint"],
        "fixed_test_manifest": protocol["fixed_test_records"][dataset],
        "source_store": protocol["source_stores"][
            (
                "knet_waveforms"
                if dataset == "knet"
                else "instancegm_waveforms"
            )
        ],
        "waveform_hash_reference": protocol[
            "selected_waveform_hash_references"
        ][dataset],
        "records_requested": len(records),
        "records_emitted": len(rows),
        "technical_failures": sum(
            int(row["technical_failure"]) for row in rows
        ),
        "waveform_identity_unverified_technical_failures": sum(
            int(
                bool(row["technical_failure"])
                and not row["waveform_sha256"]
            )
            for row in rows
        ),
        "device": "cpu",
        "torch_threads": int(protocol["execution"]["torch_threads"]),
        "workers": 1,
        "output_fields": OUTPUT_FIELDS,
        "runtime_seconds": time.time() - started,
        "worker_max_resident_set_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "records_artifact": records_artifact,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_once(attempt_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
