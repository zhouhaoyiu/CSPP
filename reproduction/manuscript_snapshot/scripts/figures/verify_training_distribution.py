#!/usr/bin/env python3
"""Independently verify MATLAB-derived figure tables and delivered artifacts."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader
from scipy.io import loadmat

from prepare_training_distribution import ROOT, OUTPUT, artifact


def main():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    summary = json.loads((OUTPUT / "distribution_summary.json").read_text())
    figures = json.loads((OUTPUT / "figure_manifest.json").read_text())
    verified_sources = []
    for recorded in [*protocol["sources"].values(), protocol["indices"], *protocol["code"]]:
        current = artifact(recorded["path"])
        assert current == recorded, recorded["path"]
        verified_sources.append(current)
    with (OUTPUT / "selected_training_records.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    with (OUTPUT / "unique_training_events.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    indices = np.load(protocol["sources"]["selected_indices"]["path"])
    matlab_indices = loadmat(OUTPUT / "selected_indices.mat")
    keys = {"InstanceGM": "ig_train", "K-NET": "knet_train", "CWA": "cwa2020_opened_train"}
    assert len(records) == summary["total_records"] == 2500
    checks = []
    for dataset in summary["datasets"]:
        name, slug = dataset["dataset"], dataset["slug"]
        rows = [row for row in records if row["dataset"] == name]
        unique = [row for row in events if row["dataset"] == name]
        selected = np.array([int(row["source_index"]) for row in rows])
        np.testing.assert_array_equal(selected, indices[keys[name]])
        np.testing.assert_array_equal(selected, matlab_indices[keys[name]].ravel())
        assert len(rows) == dataset["record_count"]
        assert len({row["event_id"] for row in rows}) == len(unique) == dataset["event_count"]
        assert len({row["station_id"] for row in rows}) == dataset["station_count"]
        assert sum(int(row["record_count"]) for row in unique) == len(rows)
        for row in unique:
            matches = [record for record in rows if record["event_id"] == row["event_id"]]
            assert len(matches) == int(row["record_count"])
            for field in ("longitude", "latitude", "magnitude", "depth_km"):
                np.testing.assert_allclose([float(record[field]) for record in matches], float(row[field]), atol=1e-6)
        points = np.loadtxt(OUTPUT / f"{slug}_epicenters.txt")
        np.testing.assert_allclose(points, [[float(row["longitude"]), float(row["latitude"])] for row in unique])
        assert len(points) == dataset["mapped_events"]
        for field in ("magnitude", "depth_km"):
            values = np.array([float(row[field]) for row in rows])
            stats = dataset[field]
            assert np.isfinite(values).all() and stats["missing_records"] == 0
            np.testing.assert_allclose([values.min(), values.max(), values.mean(), np.median(values)],
                                       [stats["minimum"], stats["maximum"], stats["mean"], stats["median"]], atol=1e-10)
            counts, _ = np.histogram(values, bins=stats["edges"])
            np.testing.assert_array_equal(counts, stats["counts"])
            plotted = np.loadtxt(OUTPUT / f"{slug}_{field}_histogram.txt")
            np.testing.assert_array_equal(plotted[:, 1], counts)
            np.testing.assert_allclose(plotted[:, 0], (np.array(stats["edges"][:-1])+stats["edges"][1:])/2)
            assert counts.sum() == stats["valid_records"] == len(rows)
        checks.append({"dataset": name, "records": len(rows), "events": len(unique), "checks": "PASS"})
    for recorded in [*figures["outputs"], figures["source"], figures["summary"]]:
        assert artifact(recorded["path"]) == recorded
    pdf = ROOT / "output/pdf/cspp_training_distribution_all.pdf"
    reader = PdfReader(pdf)
    assert len(reader.pages) == 1
    page = reader.pages[0]
    content = page.extract_text()
    for expected in ("InstanceGM", "K-NET", "CWA", "581", "612", "78", "3.40", "4.35", "5.23"):
        assert expected in content, expected
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    assert height > width*1.3 and 450 < width < 650
    images = []
    for item in figures["outputs"]:
        if Path(item["path"]).suffix != ".png":
            continue
        with Image.open(item["path"]) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert pixels.std() > 20 and (pixels.min(axis=2) < 240).mean() > .15
            images.append({"path": item["path"], "width": image.width, "height": image.height})
    report = {"status": "PASS", "source_hashes": verified_sources, "cohorts": checks,
              "histogram_check": "Independent NumPy counts against exact MATLAB-exported edges",
              "pdf": {"pages": 1, "width_pt": width, "height_pt": height, "sha256": artifact(pdf)["sha256"]},
              "pngs": images, "verifier": artifact(Path(__file__)),
              "visual_review": "Separate rendered-page inspection is required; arithmetic checks do not establish visual quality"}
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": "PASS", "cohorts": checks, "pdf_dimensions_pt": [width, height]}, indent=2))


if __name__ == "__main__":
    main()
