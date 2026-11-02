#!/usr/bin/env python3
"""Score sealed PhaseNet/EQTransformer endpoints on fixed event records."""
from __future__ import annotations

import argparse
import csv
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from scripts.evaluation import (  # noqa: E402
    eval_cspp_v27c_fixed_retrospective as retrospective,
)
from scripts.evaluation import (  # noqa: E402
    freeze_v27c_fixed_public_baseline_protocol as freeze,
)
from scripts.evaluation import v27_runner_lease as runner_lease  # noqa: E402
import eval_v24_fixed_test_models as legacy  # noqa: E402
from v20c_public_window_baselines import (  # noqa: E402
    load_public_model,
)


DEFAULT_PROTOCOL = freeze.DEFAULT_OUTPUT
OUTPUT_FIELDS = legacy.OUTPUT_FIELDS


def software_versions() -> dict[str, str]:
    return {
        package: version(package)
        for package in ("numpy", "obspy", "seisbench", "torch")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--dataset", choices=("knet", "instancegm"), required=True
    )
    parser.add_argument(
        "--model",
        choices=("phasenet_instance", "eqtransformer_original"),
        required=True,
    )
    parser.add_argument("--runner-lease", type=Path, required=True)
    parser.add_argument("--attempt-output-root", type=Path, required=True)
    return parser.parse_args()


def _same_artifact(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return (
        Path(str(left["path"])).resolve() == Path(str(right["path"])).resolve()
        and int(left["bytes"]) == int(right["bytes"])
        and str(left.get("sha256")) == str(right.get("sha256"))
    )


def load_protocol(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    protocol_path = retrospective.reject_forbidden_path(path)
    actual_sha256 = retrospective.sha256_file(protocol_path)
    if actual_sha256 != expected_sha256.lower():
        raise RuntimeError("Public-baseline protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != freeze.PROTOCOL_ID:
        raise RuntimeError("Unexpected public-baseline protocol")
    execution = protocol.get("execution", {})
    if (
        execution.get("device") not in {"cpu", "mps"}
        or int(execution.get("workers", -1)) != 1
        or int(execution.get("torch_threads", -1)) != 1
        or not bool(execution.get("fresh_process_per_unit"))
    ):
        raise RuntimeError("Public-baseline execution contract differs")
    if Path(execution.get("allowed_output_root", "")).resolve() != (
        freeze.WORK_ROOT
    ):
        raise RuntimeError("Unexpected public-baseline output root")
    freeze.require_external_output(execution["output_root"])
    freeze.require_external_output(execution["global_lock_path"])

    for relative, expected in protocol["code_sha256"].items():
        source = ROOT / relative
        if (
            not source.is_file()
            or retrospective.sha256_file(source) != str(expected)
        ):
            raise RuntimeError(f"Frozen baseline code drifted: {relative}")

    retro_path = retrospective.verify_artifact(
        protocol["retrospective_protocol"],
        require_sha256=True,
        require_mtime=False,
    )
    amendment = protocol.get("execution_amendment")
    if amendment is None:
        retro, retro_sha256 = retrospective.load_protocol(
            retro_path,
            expected_sha256=protocol["retrospective_protocol_sha256"],
        )
    else:
        retro, retro_sha256 = (
            freeze.load_retrospective_for_execution_amendment(
                retro_path,
                expected_sha256=protocol[
                    "retrospective_protocol_sha256"
                ],
                amendment=amendment,
            )
        )
    if retro_sha256 != protocol["retrospective_protocol_sha256"]:
        raise RuntimeError("Nested retrospective protocol differs")
    for field in (
        "fixed_test_records",
        "source_stores",
        "selected_waveform_hash_references",
    ):
        if protocol[field] != retro[field]:
            raise RuntimeError(
                f"Public and retrospective protocols differ: {field}"
            )
    for spec in protocol["fixed_test_records"].values():
        retrospective.verify_artifact(
            spec, require_sha256=True, require_mtime=False
        )
    for spec in protocol["source_stores"].values():
        retrospective.verify_artifact(
            spec, require_sha256=False, require_mtime=True
        )
    for spec in protocol["selected_waveform_hash_references"].values():
        retrospective.verify_artifact(
            spec, require_sha256=True, require_mtime=False
        )
    if set(protocol["public_models"]) != freeze.EXPECTED_MODELS:
        raise RuntimeError("Public model set differs")
    for model_key, spec in protocol["public_models"].items():
        methods = {
            str(point["id"]) for point in spec["operating_points"]
        }
        if methods != freeze.EXPECTED_METHODS[model_key]:
            raise RuntimeError(f"Public methods differ: {model_key}")
        for artifact_spec in spec["artifacts"].values():
            retrospective.verify_artifact(
                artifact_spec,
                require_sha256=True,
                require_mtime=False,
            )
    actual_versions = software_versions()
    if actual_versions != protocol["software_versions"]:
        raise RuntimeError(
            "Public-baseline software versions differ: "
            f"expected={protocol['software_versions']}, "
            f"actual={actual_versions}"
        )
    return protocol, actual_sha256


def load_sealed_public_model(
    spec: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    import seisbench.models as sbm

    classes = {
        "PhaseNet": sbm.PhaseNet,
        "EQTransformer": sbm.EQTransformer,
    }
    model_class = classes[str(spec["class"])]
    resolved_weights, resolved_metadata = model_class._pretrained_path(
        str(spec["weights_name"]),
        str(spec["weights_version"]),
    )
    expected_weights = Path(spec["artifacts"]["weights"]["path"]).resolve()
    expected_metadata = Path(spec["artifacts"]["metadata"]["path"]).resolve()
    if (
        resolved_weights.resolve() != expected_weights
        or resolved_metadata.resolve() != expected_metadata
    ):
        raise RuntimeError(
            "SeisBench public-model cache resolution differs from the "
            "frozen artifact paths"
        )
    model = load_public_model(spec, device=device)
    loaded_digest = freeze.tensor_state_digest(model.state_dict())
    if loaded_digest != str(spec["loaded_state_dict_sha256"]):
        raise RuntimeError(
            "Loaded public-model tensors differ from the frozen weight file"
        )
    return model, loaded_digest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def waveform_hashes(
    protocol: Mapping[str, Any],
    dataset: str,
) -> dict[tuple[str, str, str], str]:
    path = retrospective.verify_artifact(
        protocol["selected_waveform_hash_references"][dataset],
        require_sha256=True,
        require_mtime=False,
    )
    hashes_by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in read_csv(path):
        if row["dataset"] != dataset:
            raise RuntimeError("Waveform-hash reference dataset differs")
        key = (row["source_index"], row["trace_name"], row["event_id"])
        digest = row["waveform_sha256"]
        if (
            len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise RuntimeError("Waveform-hash reference is not one-to-one")
        hashes_by_key.setdefault(key, set()).add(digest)
    if any(len(values) != 1 for values in hashes_by_key.values()):
        raise RuntimeError("Waveform-hash reference is not one-to-one")
    return {
        key: next(iter(values))
        for key, values in hashes_by_key.items()
    }


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


def run_root(
    protocol: Mapping[str, Any],
    dataset: str,
    model: str,
) -> Path:
    return (
        freeze.require_external_output(protocol["execution"]["output_root"])
        / dataset
        / model
    )


def method_rows(
    source: dict[str, str],
    *,
    dataset: str,
    model: str,
    wave: np.ndarray,
    methods: Mapping[str, list[dict[str, Any]]],
    technical_error: str = "",
) -> list[dict[str, Any]]:
    return [
        legacy.scored_row(
            source,
            dataset=dataset,
            model_key=model,
            method_id=method_id,
            comparison_mode="released_full_record_annotation",
            availability_comparable=False,
            wave=wave,
            candidates=candidates,
            technical_error=technical_error,
        )
        for method_id, candidates in methods.items()
    ]


def main() -> None:
    args = parse_args()
    protocol, protocol_sha256 = load_protocol(
        args.protocol,
        expected_sha256=args.expected_protocol_sha256,
    )
    device_name = str(protocol["execution"]["device"])
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.set_num_threads(int(protocol["execution"]["torch_threads"]))
    run_path = run_root(protocol, args.dataset, args.model)
    if run_path.exists():
        raise FileExistsError(run_path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path = retrospective.require_output_path(
        args.attempt_output_root,
        Path(protocol["execution"]["allowed_output_root"]),
    )
    if (
        attempt_path.parent != run_path.parent
        or not attempt_path.name.startswith(f".{run_path.name}.attempt-")
        or not attempt_path.is_dir()
    ):
        raise RuntimeError("Runner attempt directory differs")
    runner_lease.verify_lease(
        args.runner_lease,
        protocol_sha256=protocol_sha256,
        unit={"dataset": args.dataset, "model": args.model},
        evaluator_script=Path(__file__).resolve(),
        attempt_root=attempt_path,
        published_root=run_path,
    )
    unexpected = [
        path.name
        for path in attempt_path.iterdir()
        if path.name != "runner_lease.json"
        and not path.name.startswith("._")
    ]
    if unexpected:
        raise RuntimeError(
            f"Runner attempt directory is not empty: {unexpected}"
        )

    manifest_path = retrospective.verify_artifact(
        protocol["fixed_test_records"][args.dataset],
        require_sha256=True,
        require_mtime=False,
    )
    records = read_csv(manifest_path)
    expected_count = int(
        protocol["fixed_test_cohorts"][args.dataset]["records"]
    )
    if len(records) != expected_count:
        raise RuntimeError("Fixed-record count differs")
    expected_hashes = waveform_hashes(protocol, args.dataset)
    manifest_keys = {
        (row["source_index"], row["trace_name"], row["event_id"])
        for row in records
    }
    if set(expected_hashes) != manifest_keys:
        raise RuntimeError("Waveform-hash reference does not match manifest")

    reader = legacy.WaveReader(args.dataset, protocol)
    device = torch.device(device_name)
    model, loaded_state_digest = load_sealed_public_model(
        protocol["public_models"][args.model],
        device=device,
    )
    output_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for serial, source in enumerate(records, start=1):
            wave = reader.read(source)
            key = (
                source["source_index"],
                source["trace_name"],
                source["event_id"],
            )
            digest = legacy.waveform_digest(wave)
            if digest != expected_hashes[key]:
                raise RuntimeError(
                    f"Selected waveform identity differs: {key}"
                )
            if float(source["p_arrival_sample"]) >= wave.shape[1]:
                raise RuntimeError("Reference P lies outside waveform")
            try:
                methods = legacy.score_public(
                    wave,
                    protocol,
                    args.model,
                    model,
                )
                error_text = ""
            except Exception as error:  # technical failure is a declared miss
                methods = {
                    method_id: []
                    for method_id in freeze.EXPECTED_METHODS[args.model]
                }
                error_text = f"{type(error).__name__}: {error}"
            output_rows.extend(
                method_rows(
                    source,
                    dataset=args.dataset,
                    model=args.model,
                    wave=wave,
                    methods=methods,
                    technical_error=error_text,
                )
            )
            if serial % 25 == 0 or serial == len(records):
                print(
                    f"{args.dataset} {args.model} "
                    f"[{serial}/{len(records)}]",
                    flush=True,
                )
                gc.collect()
                if device_name == "mps":
                    torch.mps.empty_cache()
    finally:
        reader.close()

    attempt_records_path = attempt_path / "records.csv"
    write_csv_once(attempt_records_path, output_rows)
    final_records_path = run_path / "records.csv"
    records_artifact = {
        **retrospective.artifact(attempt_records_path),
        "path": str(final_records_path.resolve()),
    }
    expected_methods = freeze.EXPECTED_METHODS[args.model]
    actual_versions = software_versions()
    summary = {
        "summary_id": "cspp_v27c_fixed_public_baseline_run_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "retrospective_protocol_sha256": protocol[
            "retrospective_protocol_sha256"
        ],
        "dataset": args.dataset,
        "model_key": args.model,
        "methods": sorted(expected_methods),
        "records_requested": len(records),
        "output_rows": len(output_rows),
        "technical_failure_records": len(
            {
                (row["source_index"], row["trace_name"], row["event_id"])
                for row in output_rows
                if row["technical_error"]
            }
        ),
        "waveform_identity_verified_records": len(records),
        "comparison_mode": "released_full_record_annotation",
        "availability_comparable": False,
        "device": device_name,
        "torch_threads": torch.get_num_threads(),
        "software_versions": actual_versions,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "loaded_state_dict_sha256": loaded_state_digest,
        "resolved_weight_artifact": protocol["public_models"][args.model][
            "artifacts"
        ]["weights"],
        "resolved_metadata_artifact": protocol["public_models"][args.model][
            "artifacts"
        ]["metadata"],
        "runtime_seconds": time.time() - started,
        "worker_max_resident_set_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "records_artifact": records_artifact,
    }
    attempt_summary_path = attempt_path / "summary.json"
    temporary = attempt_summary_path.with_name(
        f".{attempt_summary_path.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(attempt_summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
