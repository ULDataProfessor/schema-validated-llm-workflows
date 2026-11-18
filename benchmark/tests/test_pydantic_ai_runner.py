import asyncio
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameworks.pydantic_ai.runner import (  # noqa: E402
    PydanticAIRunner,
    _pydantic_ai_model_name,
    _usage_from_result,
)


def test_pydantic_ai_model_name_uses_chat_provider_for_openai_prefix():
    assert _pydantic_ai_model_name("openai:gpt-4o-mini") == "openai-chat:gpt-4o-mini"
    assert _pydantic_ai_model_name("anthropic:claude-sonnet-4") == "anthropic:claude-sonnet-4"


def test_pydantic_ai_runner_reports_missing_package_cleanly(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pydantic_ai" or name.startswith("pydantic_ai."):
            raise ImportError("simulated missing pydantic_ai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    runner = PydanticAIRunner()
    result = asyncio.run(
        runner.run_extraction(
            "Vendor: Acme\nInvoice Number: INV-1\nDate: 2024-01-01\nSubtotal: $1.00\nTax: $0.00\nTOTAL DUE: $1.00 USD"
        )
    )

    assert result.error is not None
    assert "pydantic-ai package not installed" in result.error


def test_pydantic_ai_usage_from_property_object():
    class Usage:
        request_tokens = 11
        response_tokens = 7

    class Result:
        usage = Usage()

    assert _usage_from_result(Result()) == (11, 7)


def test_pydantic_ai_usage_from_callable_property_object_does_not_call():
    class Usage:
        request_tokens = 13
        response_tokens = 2

        def __call__(self):
            raise AssertionError("callable usage property should not be invoked")

    class Result:
        usage = Usage()

    assert _usage_from_result(Result()) == (13, 2)


def test_pydantic_ai_usage_from_legacy_method_dict():
    class Result:
        def usage(self):
            return {"prompt_tokens": 5, "completion_tokens": 3}

    assert _usage_from_result(Result()) == (5, 3)
