"""Run the recorded final verifier-training stage after explicit data-path remapping."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_KEYS = (
    "knet_dir",
    "continuous_noise_hdf5",
    "continuous_catalog_screen",
    "hard_negative_review_manifest",
    "target_hard_negative_hdf5",
    "target_hard_negative_review_manifest",
    "target_hard_negative_catalog_screen",
    "target_background_hdf5",
    "target_background_catalog_screen",
    "cwa_opened_packet_store",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, choices=(20261101, 20261102, 20261103), default=20261101
    )
    parser.add_argument(
        "--paths",
        type=Path,
        help="JSON mapping each required data key to a local file/directory",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start training; default is a read-only preflight",
    )
    args = parser.parse_args()
    config = json.loads(
        (
            ROOT / f"reproduction/training/seed{args.seed}/training_config.json"
        ).read_text()
    )
    mapping = json.loads(args.paths.read_text()) if args.paths else {}
    missing = [
        key
        for key in DATA_KEYS
        if key not in mapping or not Path(mapping[key]).exists()
    ]
    report = {
        "seed": args.seed,
        "required_data_keys": list(DATA_KEYS),
        "missing_data_keys": missing,
        "additional_prerequisite": "SeisBench InstanceGM must be available in its configured local cache",
        "scope": "Final verifier-training stage from released pretrained candidate core; not from-scratch reproduction",
        "recorded_device": config["device"],
        "requested_device": args.device,
        "training_started": False,
    }
    print(json.dumps(report, indent=2))
    if not args.run:
        return
    if missing or args.output is None:
        parser.error(
            "--run requires --output and an existing path for every required data key"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    script = (
        ROOT
        / "reproduction/training_snapshot/scripts/training/train_domain_robust_v5.py"
    )
    spec = importlib.util.spec_from_file_location("release_training", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_argv = sys.argv
    try:
        sys.argv = [str(script)]
        namespace = module.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in config.items():
        if isinstance(getattr(namespace, key, None), Path) and value is not None:
            value = Path(value)
        setattr(namespace, key, value)
    for key in DATA_KEYS:
        setattr(namespace, key, Path(mapping[key]).expanduser().resolve())
    namespace.save_dir = args.output.resolve()
    namespace.init_checkpoint = ROOT / "reproduction/initializer.pt"
    namespace.selection_reference_npz = (
        ROOT / f"reproduction/training/seed{args.seed}/selected_indices.npz"
    )
    namespace.device = args.device
    module.parse_args = lambda: namespace
    module.main()


if __name__ == "__main__":
    main()
