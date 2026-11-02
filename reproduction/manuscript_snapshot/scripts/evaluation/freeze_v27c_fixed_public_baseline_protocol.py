#!/usr/bin/env python3
"""Freeze provenance for PhaseNet/EQTransformer fixed-record baselines."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import (  # noqa: E402
    eval_cspp_v27c_fixed_retrospective as retrospective,
)


WORK_ROOT = Path("workspace").resolve()
FORBIDDEN_VOLUME = Path("external_volume").resolve()
PROTOCOL_ID = "cspp_v27c_fixed_public_baselines_v1"
RETROSPECTIVE_PROTOCOL_ID = "cspp_v27c_fixed_retrospective_benchmark_v1"
V24_PROTOCOL_ID = "domain_robust_v24_station_level_external_test_v1"
DEFAULT_RETROSPECTIVE_PROTOCOL = (
    WORK_ROOT
    / "protocol_freeze/cspp_v27c_fixed_retrospective_protocol_20260729.json"
)
DEFAULT_OUTPUT = (
    WORK_ROOT
    / "protocol_freeze/cspp_v27c_fixed_public_baseline_protocol_20260729.json"
)
DEFAULT_RUN_ROOT = (
    WORK_ROOT
    / "evaluation/cspp_v27c_fixed_public_baselines_20260729/runs"
)
EXPECTED_MODELS = {"phasenet_instance", "eqtransformer_original"}
EXPECTED_METHODS = {
    "phasenet_instance": {"phasenet_instance_default"},
    "eqtransformer_original": {
        "eqtransformer_original_p_default",
        "eqtransformer_original_p_det_default",
    },
}
AMENDED_PUBLIC_CODE_PATHS = (
    "scripts/evaluation/freeze_v27c_fixed_public_baseline_protocol.py",
    "scripts/evaluation/eval_v27c_fixed_public_baselines.py",
    "scripts/evaluation/run_v27c_fixed_public_baselines.py",
)
CODE_PATHS = (
    "scripts/evaluation/freeze_v27c_fixed_public_baseline_protocol.py",
    "scripts/evaluation/analyze_v27_geonet_temporal_selection.py",
    "scripts/evaluation/build_v20c_geonet_score_cache.py",
    "scripts/evaluation/eval_cspp_v27c_fixed_retrospective.py",
    "scripts/evaluation/eval_v27c_fixed_public_baselines.py",
    "scripts/evaluation/run_v27c_fixed_public_baselines.py",
    "scripts/evaluation/v27_runner_lease.py",
    "scripts/evaluation/eval_v24_fixed_test_models.py",
    "scripts/evaluation/run_v20c_geonet_score_cache_safe.py",
    "src/causal_fir_decimator.py",
    "src/causal_packet_veto_features.py",
    "src/causal_stream_veto_features.py",
    "src/config.py",
    "src/cspp_v27_reused_core_score_cache.py",
    "src/current_packet_artifact_features.py",
    "src/current_packet_artifact_head.py",
    "src/data_streaming.py",
    "src/domain_robust_model_io.py",
    "src/gappy_strong_motion_stream.py",
    "src/gappy_strong_motion_stream_v2.py",
    "src/model_causal_classical_v1.py",
    "src/model_cspp_v25.py",
    "src/model_cspp_v26.py",
    "src/model_cspp_v27.py",
    "src/model_domain_robust_v10.py",
    "src/model_domain_robust_v11.py",
    "src/model_domain_robust_v5.py",
    "src/model_domain_robust_v6.py",
    "src/model_domain_robust_v7.py",
    "src/model_domain_robust_v8.py",
    "src/model_domain_robust_v9.py",
    "src/model_packet_once_v1.py",
    "src/oriented_gappy_strong_motion_stream.py",
    "src/resilient_mseed.py",
    "src/v20c_public_window_baselines.py",
    "src/v15_reused_packet_features.py",
    "src/v20c_causal_score_cache.py",
    "src/v20c_event_snr.py",
    "src/stream_quality_v5.py",
    "src/v21_causal_score_cache.py",
    "src/v21_score_shape_calibrator.py",
)


def tensor_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrospective-protocol",
        type=Path,
        default=DEFAULT_RETROSPECTIVE_PROTOCOL,
    )
    parser.add_argument(
        "--expected-retrospective-protocol-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--expected-parent-protocol-sha256")
    parser.add_argument("--failed-attempt-root", type=Path)
    parser.add_argument("--superseded-protocol", type=Path)
    parser.add_argument("--expected-superseded-protocol-sha256")
    args = parser.parse_args()
    amendment_values = (
        args.parent_protocol,
        args.expected_parent_protocol_sha256,
        args.failed_attempt_root,
    )
    if any(value is not None for value in amendment_values) and not all(
        value is not None for value in amendment_values
    ):
        parser.error(
            "--parent-protocol, --expected-parent-protocol-sha256, and "
            "--failed-attempt-root must be supplied together"
        )
    superseded_values = (
        args.superseded_protocol,
        args.expected_superseded_protocol_sha256,
    )
    if any(value is not None for value in superseded_values) and not all(
        value is not None for value in superseded_values
    ):
        parser.error(
            "--superseded-protocol and "
            "--expected-superseded-protocol-sha256 must be supplied together"
        )
    if args.superseded_protocol is not None and args.parent_protocol is None:
        parser.error(
            "--superseded-protocol requires the complete parent amendment"
        )
    return args


def require_external_output(path: str | Path) -> Path:
    resolved = retrospective.reject_forbidden_path(path)
    if not retrospective.is_within(resolved, WORK_ROOT):
        raise RuntimeError(
            f"Public-baseline output must remain under {WORK_ROOT}: {resolved}"
        )
    return resolved


def _artifact_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = retrospective.verify_artifact(
        spec,
        require_sha256=bool(spec.get("sha256")),
        require_mtime=bool(spec.get("mtime_ns")),
    )
    return {
        **dict(spec),
        "path": str(path),
        "bytes": int(path.stat().st_size),
    }


def _verify_retrospective_scientific_contract(
    protocol: Mapping[str, Any],
) -> None:
    if protocol.get("protocol_id") != RETROSPECTIVE_PROTOCOL_ID:
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
        or not math.isclose(
            float(point.get("event_threshold", math.nan)),
            0.975,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("Retrospective operating point differs")
    model = protocol.get("model", {})
    if (
        model.get("selected_arm") != "C0"
        or int(model.get("primary_seed", -1)) != 20261101
        or int(model.get("ensemble_members", 0)) != 1
    ):
        raise RuntimeError("Retrospective model selection differs")
    execution = protocol.get("execution", {})
    if (
        execution.get("device") != "cpu"
        or int(execution.get("workers", 0)) != 1
        or int(execution.get("torch_threads", 0)) != 1
    ):
        raise RuntimeError("Retrospective execution contract differs")
    allowed_root = Path(execution["allowed_output_root"]).resolve()
    if allowed_root != retrospective.ALLOWED_OUTPUT_ROOT:
        raise RuntimeError("Unexpected retrospective output root")
    retrospective.require_output_path(
        execution["score_output_root"], allowed_root
    )
    retrospective.require_output_path(
        execution["analysis_output_root"], allowed_root
    )

    for key in (
        "selection_freeze",
        "development_protocol",
        "v24_fixed_test_protocol",
    ):
        retrospective.verify_artifact(
            protocol[key],
            require_sha256=True,
            require_mtime=False,
        )
    selection_result = protocol.get("selection_result")
    if selection_result:
        retrospective.verify_artifact(
            selection_result,
            require_sha256=True,
            require_mtime=False,
        )
    retrospective.verify_artifact(
        model["checkpoint"],
        require_sha256=True,
        require_mtime=False,
    )
    retrospective.verify_artifact(
        model["training_selection_reference"],
        require_sha256=True,
        require_mtime=False,
    )
    for spec in protocol["fixed_test_records"].values():
        retrospective.verify_artifact(
            spec,
            require_sha256=True,
            require_mtime=False,
        )
    for spec in protocol["source_stores"].values():
        retrospective.verify_artifact(
            spec,
            require_sha256=False,
            require_mtime=True,
        )
    references = protocol.get("selected_waveform_hash_references", {})
    if set(references) != {"knet", "instancegm"}:
        raise RuntimeError("Waveform-hash references are incomplete")
    for spec in references.values():
        retrospective.verify_artifact(
            spec,
            require_sha256=True,
            require_mtime=False,
        )
    overlap = protocol.get("training_test_overlap_audit", {})
    overlap_datasets = overlap.get("datasets", {})
    if (
        overlap.get("status") != "zero_source_index_intersection"
        or set(overlap_datasets) != {"knet", "instancegm"}
    ):
        raise RuntimeError("Training/test overlap audit differs")
    for entries in overlap_datasets.values():
        if not entries or any(
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


def load_retrospective_for_execution_amendment(
    path: Path,
    *,
    expected_sha256: str,
    amendment: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    protocol_path = retrospective.reject_forbidden_path(path)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    actual_hash = retrospective.sha256_file(protocol_path)
    if actual_hash != expected_sha256.lower():
        raise RuntimeError("Retrospective protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    parent_path = retrospective.verify_artifact(
        amendment["parent_protocol"],
        require_sha256=True,
        require_mtime=False,
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("protocol_id") != PROTOCOL_ID
        or parent.get("retrospective_protocol_sha256") != actual_hash
        or parent.get("retrospective_protocol", {}).get("sha256")
        != actual_hash
    ):
        raise RuntimeError(
            "Parent public protocol does not bind this retrospective protocol"
        )

    transitions = amendment.get("code_sha256_transitions", {})
    if set(transitions) != set(AMENDED_PUBLIC_CODE_PATHS):
        raise RuntimeError("Public amendment code transition set differs")
    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        current = retrospective.sha256_file(source)
        if relative not in transitions:
            if current != str(expected):
                raise RuntimeError(
                    f"Frozen evaluation code drifted: {relative}"
                )
            continue
        transition = transitions[relative]
        if (
            transition.get("parent_sha256") != str(expected)
            or parent.get("code_sha256", {}).get(relative) != str(expected)
            or transition.get("amended_sha256") != current
            or current == str(expected)
        ):
            raise RuntimeError(
                f"Public amendment code identity differs: {relative}"
            )
    if not set(transitions) <= set(protocol["code_sha256"]):
        raise RuntimeError(
            "Retrospective protocol did not bind every amended code path"
        )
    _verify_retrospective_scientific_contract(protocol)
    return protocol, actual_hash


def execution_amendment(
    *,
    parent_protocol_path: Path,
    expected_parent_protocol_sha256: str,
    failed_attempt_root: Path,
    superseded_protocol_path: Path | None = None,
    expected_superseded_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    parent_path = retrospective.reject_forbidden_path(
        parent_protocol_path
    )
    if (
        not parent_path.is_file()
        or retrospective.sha256_file(parent_path)
        != expected_parent_protocol_sha256
    ):
        raise RuntimeError("Parent public-baseline protocol identity differs")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected parent public-baseline protocol")
    attempt = require_external_output(failed_attempt_root)
    lease_path = attempt / "runner_lease.json"
    if (
        not attempt.is_dir()
        or not lease_path.is_file()
        or any(
            (attempt / name).exists()
            for name in (
                "summary.json",
                "records.csv",
                "execution_audit.json",
            )
        )
    ):
        raise RuntimeError("Pre-inference public-baseline failure differs")
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    unit = {"dataset": "knet", "model": "phasenet_instance"}
    parent_code_sha256 = {
        relative: parent["code_sha256"][relative]
        for relative in AMENDED_PUBLIC_CODE_PATHS
    }
    parent_runner_sha256 = parent_code_sha256[
        "scripts/evaluation/run_v27c_fixed_public_baselines.py"
    ]
    parent_evaluator_sha256 = parent_code_sha256[
        "scripts/evaluation/eval_v27c_fixed_public_baselines.py"
    ]
    expected_published = (
        Path(parent["execution"]["output_root"])
        / "knet"
        / "phasenet_instance"
    ).resolve()
    if (
        lease.get("protocol_sha256")
        != expected_parent_protocol_sha256
        or lease.get("unit") != unit
        or lease.get("runner_script_sha256")
        != parent_runner_sha256
        or lease.get("evaluator_script_sha256")
        != parent_evaluator_sha256
        or Path(str(lease.get("attempt_root", ""))).resolve() != attempt
        or Path(str(lease.get("published_root", ""))).resolve()
        != expected_published
        or expected_published.exists()
    ):
        raise RuntimeError("Failed public-baseline lease identity differs")
    current_code_sha256 = {
        relative: retrospective.sha256_file(ROOT / relative)
        for relative in AMENDED_PUBLIC_CODE_PATHS
    }
    current_runner_sha256 = current_code_sha256[
        "scripts/evaluation/run_v27c_fixed_public_baselines.py"
    ]
    current_evaluator_sha256 = current_code_sha256[
        "scripts/evaluation/eval_v27c_fixed_public_baselines.py"
    ]
    if any(
        current_code_sha256[relative] == parent_code_sha256[relative]
        for relative in AMENDED_PUBLIC_CODE_PATHS
    ):
        raise RuntimeError("Public-baseline amendment code did not change")

    superseded_record = None
    if superseded_protocol_path is not None:
        superseded_path = retrospective.reject_forbidden_path(
            superseded_protocol_path
        )
        if (
            expected_superseded_protocol_sha256 is None
            or not superseded_path.is_file()
            or retrospective.sha256_file(superseded_path)
            != expected_superseded_protocol_sha256
        ):
            raise RuntimeError("Superseded public protocol identity differs")
        superseded = json.loads(
            superseded_path.read_text(encoding="utf-8")
        )
        superseded_amendment = superseded.get("execution_amendment", {})
        superseded_output = Path(
            superseded.get("execution", {}).get("output_root", "")
        ).resolve()
        if (
            superseded.get("protocol_id") != PROTOCOL_ID
            or superseded.get("retrospective_protocol_sha256")
            != parent.get("retrospective_protocol_sha256")
            or superseded_amendment.get("parent_protocol", {}).get("sha256")
            != expected_parent_protocol_sha256
            or superseded_output.exists()
        ):
            raise RuntimeError("Superseded public protocol state differs")
        superseded_record = {
            "protocol": retrospective.artifact(superseded_path),
            "output_root": str(superseded_output),
            "output_root_absent": True,
            "failure_stage": "protocol_validation_before_model_load",
            "observed_error": (
                "Nested retrospective protocol loading still used the "
                "unamended strict code-hash verifier and rejected the "
                "already declared public evaluator drift."
            ),
            "scientific_output_created": False,
        }

    payload = {
        "amendment_id": (
            "cspp_v27c_fixed_public_baseline_execution_a3"
            if superseded_record is not None
            else "cspp_v27c_fixed_public_baseline_execution_a2"
        ),
        "parent_protocol": retrospective.artifact(parent_path),
        "failed_attempt": {
            "root": str(attempt),
            "runner_lease": retrospective.artifact(lease_path),
            "summary_absent": True,
            "records_absent": True,
            "execution_audit_absent": True,
            "published_root_absent": True,
        },
        "observed_pre_inference_failure": (
            "The historical waveform-hash reference contains three rows "
            "per record, one for each old V25 operating point. The parent "
            "public evaluator rejected any duplicate key before loading a "
            "model, although the parent retrospective protocol had already "
            "verified that all repeated hashes agree."
        ),
        "input_identity_fix": (
            "Group rows by dataset/source_index/trace_name/event_id and "
            "accept a key only when its set of waveform SHA-256 values has "
            "size one. Conflicting hashes still fail closed."
        ),
        "resource_guard_fix": (
            "Accept an unreadable child RSS sample only when a second poll "
            "proves the child exited during that sample. Live-process RSS "
            "sampling failures still terminate the unit."
        ),
        "parent_runner_sha256": parent_runner_sha256,
        "amended_runner_sha256": current_runner_sha256,
        "parent_evaluator_sha256": parent_evaluator_sha256,
        "amended_evaluator_sha256": current_evaluator_sha256,
        "code_sha256_transitions": {
            relative: {
                "parent_sha256": parent_code_sha256[relative],
                "amended_sha256": current_code_sha256[relative],
            }
            for relative in AMENDED_PUBLIC_CODE_PATHS
        },
        "scientific_configuration_changed": False,
        "model_loaded_before_failure": False,
        "previous_attempt_reused": False,
        "rerun_required": True,
        "claim_boundary": (
            "This amendment changes only input-identity normalization for "
            "repeated "
            "legacy operating-point rows and post-exit process sampling. "
            "When present, the nested-protocol validation correction routes "
            "the same declared code transitions through the same bounded "
            "verifier. "
            "Weights, records, waveforms, methods, thresholds, endpoint "
            "metrics, and dataset membership are unchanged."
        ),
    }
    if superseded_record is not None:
        payload["superseded_protocol_validation"] = superseded_record
    return payload


def build_protocol(
    *,
    retrospective_protocol_path: Path,
    expected_retrospective_protocol_sha256: str,
    output_path: Path,
    run_root: Path,
    execution_amendment_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = require_external_output(output_path)
    runs = require_external_output(run_root)
    if output.exists():
        raise FileExistsError(output)
    if runs.exists():
        raise FileExistsError(runs)

    retro_path = retrospective.reject_forbidden_path(
        retrospective_protocol_path
    )
    if execution_amendment_record is None:
        retro, retro_sha256 = retrospective.load_protocol(
            retro_path,
            expected_sha256=expected_retrospective_protocol_sha256.lower(),
        )
    else:
        retro, retro_sha256 = (
            load_retrospective_for_execution_amendment(
                retro_path,
                expected_sha256=(
                    expected_retrospective_protocol_sha256.lower()
                ),
                amendment=execution_amendment_record,
            )
        )
    if retro.get("protocol_id") != RETROSPECTIVE_PROTOCOL_ID:
        raise RuntimeError("Unexpected retrospective protocol identifier")

    v24_path = retrospective.verify_artifact(
        retro["v24_fixed_test_protocol"],
        require_sha256=True,
        require_mtime=False,
    )
    v24 = json.loads(v24_path.read_text(encoding="utf-8"))
    if v24.get("protocol_id") != V24_PROTOCOL_ID:
        raise RuntimeError("Unexpected source public-model protocol")
    models = v24.get("public_models", {})
    if set(models) != EXPECTED_MODELS:
        raise RuntimeError("Frozen public-model set differs")
    public_models: dict[str, Any] = {}
    for model_key in sorted(EXPECTED_MODELS):
        spec = dict(models[model_key])
        methods = {
            str(point["id"]) for point in spec.get("operating_points", [])
        }
        if methods != EXPECTED_METHODS[model_key]:
            raise RuntimeError(
                f"Public operating points differ for {model_key}"
            )
        artifacts = {
            key: _artifact_identity(value)
            for key, value in spec["artifacts"].items()
        }
        stored_state = torch.load(
            artifacts["weights"]["path"],
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(stored_state, Mapping) or not all(
            isinstance(value, torch.Tensor)
            for value in stored_state.values()
        ):
            raise RuntimeError(
                f"Unexpected public weight payload: {model_key}"
            )
        public_models[model_key] = {
            **spec,
            "artifacts": artifacts,
            "loaded_state_dict_sha256": tensor_state_digest(stored_state),
        }

    for relative in CODE_PATHS:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)

    payload = {
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "retrospective_protocol": retrospective.artifact(retro_path),
        "retrospective_protocol_sha256": retro_sha256,
        "fixed_test_records": retro["fixed_test_records"],
        "fixed_test_cohorts": retro["fixed_test_cohorts"],
        "source_stores": retro["source_stores"],
        "waveform_input_contracts": retro["waveform_input_contracts"],
        "selected_waveform_hash_references": retro[
            "selected_waveform_hash_references"
        ],
        "public_models": public_models,
        "endpoint": {
            "evaluation_unit": "one labeled three-component event record",
            "interface": "released SeisBench full-record annotation",
            "match_tolerance_seconds": 1.5,
            "hit_windows_seconds": [0.5, 1.0, 1.5],
            "candidate_accounting": (
                "One candidate within +/-1.5 s is a hit; additional "
                "candidates are extra candidates within the event record."
            ),
            "availability_latency_comparison_permitted": False,
            "technical_failure_policy": "count_as_miss",
            "test_set_threshold_tuning_permitted": False,
        },
        "execution": {
            "device": "mps",
            "torch_threads": 1,
            "workers": 1,
            "fresh_process_per_unit": True,
            "output_root": str(runs),
            "allowed_output_root": str(WORK_ROOT),
            "forbidden_volume": str(FORBIDDEN_VOLUME),
            "global_lock_path": str(
                WORK_ROOT / "locks/cspp_v27_heavy_evaluation.lock"
            ),
            "exclusive_process_patterns": [
                "run_v27_geonet_temporal_score_workers.py",
                "build_v27_geonet_temporal_score_cache.py",
                "run_v27_geonet_final_score_workers.py",
                "build_v27_geonet_final_score_cache.py",
                "run_cspp_v27c_fixed_retrospective.py",
                "eval_cspp_v27c_fixed_retrospective.py",
                "run_v27c_fixed_public_baselines.py",
                "eval_v27c_fixed_public_baselines.py",
            ],
            "maximum_worker_resident_set_gib": 8.0,
            "minimum_system_available_memory_gib": 8.0,
            "maximum_system_swap_used_gib": 4.0,
            "memory_guard_poll_seconds": 0.25,
        },
        "software_versions": {
            package: version(package)
            for package in ("numpy", "obspy", "seisbench", "torch")
        },
        "code_sha256": {
            relative: retrospective.sha256_file(ROOT / relative)
            for relative in CODE_PATHS
        },
        "claim_boundary": {
            "comparison": (
                "Same fixed K-NET/InstanceGM record manifests and released "
                "public weights/operating points; no fixed-test tuning."
            ),
            "causality": (
                "The public models use their released full-record annotation "
                "interfaces. Their endpoint accuracy is comparable, but "
                "their availability latency is not compared with CSPP."
            ),
            "test_status": (
                "K-NET and InstanceGM are historically inspected fixed "
                "retrospective partitions, not new blind tests."
            ),
        },
    }
    if execution_amendment_record is not None:
        payload["execution_amendment"] = dict(
            execution_amendment_record
        )
    return payload


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    amendment = None
    if args.parent_protocol is not None:
        amendment = execution_amendment(
            parent_protocol_path=args.parent_protocol,
            expected_parent_protocol_sha256=(
                args.expected_parent_protocol_sha256
            ),
            failed_attempt_root=args.failed_attempt_root,
            superseded_protocol_path=args.superseded_protocol,
            expected_superseded_protocol_sha256=(
                args.expected_superseded_protocol_sha256
            ),
        )
    payload = build_protocol(
        retrospective_protocol_path=args.retrospective_protocol,
        expected_retrospective_protocol_sha256=(
            args.expected_retrospective_protocol_sha256
        ),
        output_path=args.output,
        run_root=args.run_root,
        execution_amendment_record=amendment,
    )
    output = require_external_output(args.output)
    write_json_once(output, payload)
    print(
        json.dumps(
            retrospective.artifact(output),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
