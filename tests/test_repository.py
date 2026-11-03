import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_release import audit_public_files, distributed_files, verify


def test_declared_checkpoints_and_licenses_present():
    models = json.loads((ROOT / "models/model_manifest.json").read_text())
    assert {item["seed"] for item in models["checkpoints"]} == {20261101, 20261102, 20261103}
    for item in models["checkpoints"]:
        assert (ROOT / item["public_path"]).is_file()
    assert (ROOT / "reproduction/initializer.pt").is_file()
    for name in ("LICENSE", "MODEL_LICENSE.md"):
        assert "MIT" in (ROOT / name).read_text()


def test_restricted_data_and_access_promises_absent():
    files = distributed_files(ROOT)
    audit_public_files(ROOT, files)
    for relative in ("DATA_ACCESS.md", "README.md"):
        text = (ROOT / relative).read_text()
        assert "editors" in text and "reviewers" in text
        assert "No request-based access route" in text


def test_unlisted_raw_file_fails_release_verification(tmp_path):
    (tmp_path / "private.h5").write_bytes(b"not a dataset")
    with pytest.raises(ValueError, match="private"):
        verify(tmp_path)
    with pytest.raises(ValueError, match="private"):
        audit_public_files(tmp_path, distributed_files(tmp_path))


def test_symlink_is_not_distributed(tmp_path):
    (tmp_path / "target.txt").write_text("example")
    (tmp_path / "linked.txt").symlink_to(tmp_path / "target.txt")
    with pytest.raises(ValueError, match="symbolic"):
        distributed_files(tmp_path)
