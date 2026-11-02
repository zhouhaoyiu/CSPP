import importlib.util
from pathlib import Path


def test_public_source_data():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("verify_source_data", root / "tools/verify_source_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify()
    assert result["training_records"] == 2500
    assert result["distribution_cohorts"] == 8
