"""Verify the public source-data inventory and recovered figure aggregates."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence/source_data"


def rows(name):
    with (DATA / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify():
    manifest = json.loads((DATA / "manifest.json").read_text())
    for item in manifest["files"]:
        path = DATA / item["path"]
        assert DATA.resolve() in path.resolve().parents
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
    expected = {row["cohort"]: row for row in rows("Distribution_summary_verified.csv")}
    training = rows("Figure_5_training_records.csv")
    groups = {name: [r for r in training if r["dataset"] == name] for name in ("InstanceGM", "K-NET", "CWA")}
    for name in ("geonet_arm", "geonet_calibration", "geonet_ablation", "texnet", "ncsn_original"):
        figure = "S1" if name.startswith("geonet_") else "S2"
        groups[name] = rows(f"Figure_{figure}_{name}_records.csv")
    for name, records in groups.items():
        counts = (len(records), len({r["event_id"] for r in records}),
                  len({".".join(r["station_id"].split(".")[:2]) if name.startswith("geonet_") else r["station_id"] for r in records}))
        assert counts == tuple(int(expected[name][key]) for key in ("records", "events", "stations")), name
        for key in ("magnitude", "depth_km", "epicentral_distance_km"):
            mean = math.fsum(float(r[key]) for r in records) / len(records)
            assert math.isclose(mean, float(expected[name][key]), abs_tol=1e-10), (name, key)
    days = rows("Figure_S5_ncsn_station_days.csv")
    totals = {}
    for row in days:
        key = (row["cohort"], row["station_key"])
        totals[key] = totals.get(key, 0.0) + float(row["eligible_station_days"])
    for row in rows("Figure_S5_ncsn_station_summary.csv"):
        assert math.isclose(totals[(row["cohort"], row["station_key"])], float(row["valid_station_days"]), abs_tol=1e-10)
    geonet = rows("Figure_S5_geonet_station_days.csv")
    assert len(geonet) == 144 and {r["status"] for r in geonet} == {"accepted"}
    assert len({(r["station_key"], r["date_utc"]) for r in geonet}) == 144
    return {"verified_files": len(manifest["files"]), "distribution_cohorts": len(groups),
            "training_records": len(training), "ncsn_station_day_rows": len(days), "geonet_accepted_station_days": len(geonet),
            "scope": "Public figure inputs and saved values only; no Chinese inputs or full raw-data reproduction."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
