import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_release import audit_public_files, distributed_files, verify


def test_scientific_assets_and_model_license_unchanged():
    previous = json.loads((ROOT / "provenance/upstream_manifest.json").read_text())
    prefixes = ("models/", "evidence/", "protocols/", "reproduction/training/", "reproduction/training_snapshot/")
    for spec in previous["files"]:
        path = spec["path"]
        if path.startswith(prefixes) or path in {"reproduction/initializer.pt", "LICENSE", "MODEL_LICENSE.md"}:
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == spec["sha256"]


def test_restricted_data_and_access_promises_absent():
    files = distributed_files(ROOT)
    audit_public_files(ROOT, files)
    for relative in ("DATA_ACCESS.md", "README.md"):
        text = (ROOT / relative).read_text()
        assert "editors" in text and "reviewers" in text
        assert "No request-based access route" in text


def test_unlisted_raw_file_fails_release_verification(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": []}))
    (tmp_path / "private.h5").write_bytes(b"not a dataset")
    with pytest.raises(AssertionError, match="inventory"):
        verify(tmp_path)
    with pytest.raises(ValueError, match="private"):
        audit_public_files(tmp_path, distributed_files(tmp_path))


def test_symlink_is_not_distributed(tmp_path):
    (tmp_path / "target.txt").write_text("example")
    (tmp_path / "linked.txt").symlink_to(tmp_path / "target.txt")
    with pytest.raises(ValueError, match="symbolic"):
        distributed_files(tmp_path)
