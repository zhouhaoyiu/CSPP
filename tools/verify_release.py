"""Check the distribution for restricted data, private paths and credentials."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist", "runs"}
RAW_SUFFIXES = {".mseed", ".seed", ".sac", ".h5", ".hdf5", ".npy", ".ud", ".ns", ".ew"}
PUBLIC_DEMO_FILES = {
    "examples/data/ncsn_75079016_KSM/native.mseed",
    "examples/data/ncsn_75079016_KSM/waveform.npy",
}


def distributed_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p in SKIP or p.endswith(".egg-info") for p in relative.parts):
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        if path.is_symlink():
            raise ValueError(f"Release cannot contain symbolic links: {relative}")
        if path.is_file():
            files.append(path)
    return files


def audit_public_files(root, files):
    for path in files:
        relative = path.relative_to(root).as_posix()
        if (relative not in PUBLIC_DEMO_FILES and path.suffix.lower() in RAW_SUFFIXES) or path.name.startswith(".env"):
            raise ValueError(f"Unapproved raw or private file: {relative}")
        if path.suffix == ".npz" and not (
            relative == "tests/fixtures/synthetic_regression.npz"
            or re.fullmatch(r"reproduction/training/seed2026110[123]/selected_indices\.npz", relative)
        ):
            raise ValueError(f"Unapproved array archive: {relative}")
        if path.suffix == ".pt" and not (
            relative == "reproduction/initializer.pt"
            or re.fullmatch(r"models/cspp_seed2026110[123]\.pt", relative)
        ):
            raise ValueError(f"Unapproved checkpoint: {relative}")
        if path.suffix not in {".py", ".m", ".md", ".json", ".csv", ".txt", ".yml", ".toml", ".cff"}:
            continue
        content = path.read_text()
        if re.search(r"/(?:Users|Volumes|home)/[^\s]+", content):
            raise ValueError(f"Private absolute path: {relative}")
        if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)", content):
            raise ValueError(f"Possible credential: {relative}")


def verify(root=ROOT):
    root = Path(root)
    actual_files = distributed_files(root)
    audit_public_files(root, actual_files)
    return len(actual_files)


if __name__ == "__main__":
    print(json.dumps({"checked_files": verify(), "distribution_checks_passed": True}))
