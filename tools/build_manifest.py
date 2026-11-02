"""Regenerate the release inventory after reviewing a new repository version."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from verify_release import audit_public_files, distributed_files

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = distributed_files(ROOT)
    audit_public_files(ROOT, files)
    manifest = {
        "release": "0.2.0",
        "date": "2026-11-02",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_scope": "All release files except this manifest; excludes git, caches, builds and local runs",
        "files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(files), "privacy_checks_passed": True}))


if __name__ == "__main__":
    main()
