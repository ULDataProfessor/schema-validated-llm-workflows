import asyncio
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameworks.agno.runner import AgnoRunner, _usage_from_response  # noqa: E402


def test_agno_runner_reports_missing_package_cleanly(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("agno"):
            raise ImportError("simulated missing agno")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    runner = AgnoRunner()
    result = asyncio.run(
        runner.run_extraction(
            "Vendor: Acme\nInvoice Number: INV-1\nDate: 2024-01-01\nSubtotal: $1.00\nTax: $0.00\nTOTAL DUE: $1.00 USD"
        )
    )

    assert result.error is not None
    assert "agno package not installed" in result.error


def test_agno_usage_from_response_dict_metrics():
    class Response:
        metrics = {"input_tokens": 7, "output_tokens": 3}

    assert _usage_from_response(Response()) == (7, 3)
