import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_example():
    spec = importlib.util.spec_from_file_location("public_waveform", ROOT / "examples/public_waveform.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_waveform_expected_output():
    demo = load_example()
    summary, results = demo.run_demo()
    demo.verify(summary)
    assert len(results) == 480
    assert summary["picks"][0]["output_delay_seconds"] == pytest.approx(0.98)


def test_public_waveform_verifier_rejects_changed_count():
    import json

    demo = load_example()
    summary = json.loads((demo.DATA / "expected_output.json").read_text())
    summary["candidate_count"] = 0
    with pytest.raises(AssertionError, match="candidate_count"):
        demo.verify(summary)
