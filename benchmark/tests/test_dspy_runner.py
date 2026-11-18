import asyncio
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameworks.dspy.runner import (  # noqa: E402
    DSPyRunner,
    _dspy_model_name,
    _usage_from_prediction,
)


def test_dspy_model_name_converts_openai_prefix():
    assert _dspy_model_name("openai:gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _dspy_model_name("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"


def test_dspy_runner_reports_missing_package_cleanly(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dspy" or name.startswith("dspy."):
            raise ImportError("simulated missing dspy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    runner = DSPyRunner()
    result = asyncio.run(
        runner.run_extraction(
            "Vendor: Acme\nInvoice Number: INV-1\nDate: 2024-01-01\nSubtotal: $1.00\nTax: $0.00\nTOTAL DUE: $1.00 USD"
        )
    )

    assert result.error is not None
    assert "dspy package not installed" in result.error


def test_dspy_usage_from_nested_prediction_usage():
    class Prediction:
        def get_lm_usage(self):
            return {
                "openai/gpt-4o-mini": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                }
            }

    assert _usage_from_prediction(Prediction()) == (10, 4)
