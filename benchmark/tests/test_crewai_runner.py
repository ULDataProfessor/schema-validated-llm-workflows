import asyncio
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameworks.crewai.runner import (  # noqa: E402
    CrewAIRunner,
    _crewai_model_name,
    _structured_crew_value,
    _usage_from_crew_output,
)


def test_crewai_model_name_converts_openai_prefix():
    assert _crewai_model_name("openai:gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _crewai_model_name("gpt-4o-mini") == "gpt-4o-mini"


def test_crewai_runner_reports_missing_package_cleanly(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "crewai" or name.startswith("crewai."):
            raise ImportError("simulated missing crewai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    runner = CrewAIRunner()
    result = asyncio.run(
        runner.run_extraction(
            "Vendor: Acme\nInvoice Number: INV-1\nDate: 2024-01-01\nSubtotal: $1.00\nTax: $0.00\nTOTAL DUE: $1.00 USD"
        )
    )

    assert result.error is not None
    assert "crewai package not installed" in result.error


def test_crewai_usage_from_output_token_usage_dict():
    class Output:
        token_usage = {"prompt_tokens": 8, "completion_tokens": 5}

    assert _usage_from_crew_output(Output()) == (8, 5)


def test_crewai_structured_value_prefers_output_pydantic():
    expected = object()

    class Output:
        pydantic = expected
        json_dict = {"fallback": True}
        tasks_output = []

    assert _structured_crew_value(Output()) is expected
