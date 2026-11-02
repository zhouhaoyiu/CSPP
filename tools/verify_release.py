"""Verify every distributed file using the release manifest (standard library only)."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist", "runs"}
RAW_SUFFIXES = {".mseed", ".seed", ".sac", ".h5", ".hdf5", ".npy", ".ud", ".ns", ".ew"}
PUBLIC_DEMO_FILES = {
    "examples/data/ncsn_75079016_KSM/native.mseed": "4611435e02268363f52e9d21a6170a561312b7a60d04b91899fd2574e621a7fb",
    "examples/data/ncsn_75079016_KSM/waveform.npy": "39b3c1caf426727de9a3af296c5a76c218831d849975c412a401914c6bb73b1d",
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
        if path.is_file() and relative.as_posix() != "MANIFEST.json":
            files.append(path)
    return files


def audit_public_files(root, files):
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative in PUBLIC_DEMO_FILES:
            if hashlib.sha256(path.read_bytes()).hexdigest() != PUBLIC_DEMO_FILES[relative]:
                raise ValueError(f"Public demo hash mismatch: {relative}")
        elif path.suffix.lower() in RAW_SUFFIXES or path.name.startswith(".env"):
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
    manifest = json.loads((root / "MANIFEST.json").read_text())
    paths = [spec["path"] for spec in manifest["files"]]
    assert len(paths) == len(set(paths)), "Duplicate manifest entries"
    actual_files = distributed_files(root)
    actual_paths = {p.relative_to(root).as_posix() for p in actual_files}
    assert set(paths) == actual_paths, "Manifest differs from distributed file inventory"
    audit_public_files(root, actual_files)
    for spec in manifest["files"]:
        path = root / spec["path"]
        assert root.resolve() in path.resolve().parents, spec["path"]
        assert path.is_file(), spec["path"]
        assert path.stat().st_size == spec["bytes"], spec["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"], spec[
            "path"
        ]
    return len(manifest["files"])


if __name__ == "__main__":
    print(json.dumps({"verified_files": verify(), "all_hashes_match": True}))
