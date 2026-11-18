"""PydanticAI implementation for structured benchmark tasks."""

from __future__ import annotations

import time
from typing import Any

from data.schemas.compliance import PolicyAnswer
from data.schemas.extraction import ComplaintExtraction, InvoiceExtraction
from data.schemas.sql import SQLGenerationResult
from data.schemas.summarization import ExecutiveSummary
from data.schemas.workflow import WorkflowResult
from eval.attempt_trace import attempt_event, retries_from_attempts
from eval.metrics import execute_workflow_tool, field_accuracy
from frameworks.base import FrameworkRunner, RunResult
from tasks.extraction.prompts import get_extraction_prompt
from tasks.prompts import build_task_user_prompt, get_task_system_prompt


def _pydantic_ai_model_name(model: str) -> str:
    """Keep OpenAI runs on the chat-completions provider across PydanticAI v2."""
    if model.startswith("openai:"):
        return model.replace("openai:", "openai-chat:", 1)
    return model


def _usage_from_result(result: Any) -> tuple[int, int]:
    usage = getattr(result, "usage", None)
    tokens = _tokens_from_usage(usage)
    if tokens is not None:
        return tokens
    if callable(usage):
        return _tokens_from_usage(usage()) or (0, 0)
    return (0, 0)


def _tokens_from_usage(usage: Any) -> tuple[int, int] | None:
    if usage is None:
        return None

    if not isinstance(usage, dict) and not any(
        hasattr(usage, attr)
        for attr in (
            "request_tokens",
            "prompt_tokens",
            "input_tokens",
            "response_tokens",
            "completion_tokens",
            "output_tokens",
        )
    ):
        return None

    if isinstance(usage, dict):
        tokens_in = (
            usage.get("request_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        tokens_out = (
            usage.get("response_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        return (int(tokens_in), int(tokens_out))

    tokens_in = (
        getattr(usage, "request_tokens", None)
        or getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    tokens_out = (
        getattr(usage, "response_tokens", None)
        or getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )
    return (int(tokens_in), int(tokens_out))


class PydanticAIRunner(FrameworkRunner):
    name = "pydantic_ai"

    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        max_retries: int = 3,
        *,
        enable_validators: bool = False,
        use_typed_deps: bool = False,
    ):
        self.model = _pydantic_ai_model_name(model)
        self.max_retries = max_retries
        self.enable_validators = enable_validators
        self.use_typed_deps = use_typed_deps

    async def _run_agent(
        self,
        *,
        schema_cls: type,
        system_prompt: str,
        user_prompt: str,
    ) -> RunResult:
        start = time.perf_counter()
        raw: str | None = None
        tokens_in = tokens_out = 0
        trace_events: list[dict[str, Any]] = []
        last_error: str | None = None
        provider_calls = 0

        try:
            from pydantic_ai import Agent

            # Observe attempts in the harness loop; disable hidden Agent retries.
            agent = Agent(
                self.model,
                output_type=schema_cls,
                system_prompt=system_prompt,
                retries=1,
            )
            prompt = user_prompt
            for attempt in range(self.max_retries + 1):
                provider_calls += 1
                try:
                    result = await agent.run(prompt)
                    output = result.output
                    raw = output.model_dump_json()
                    call_in, call_out = _usage_from_result(result)
                    tokens_in += call_in or 0
                    tokens_out += call_out or 0
                    trace_events.append(
                        attempt_event(attempt=attempt, status="schema_valid")
                    )
                    elapsed = (time.perf_counter() - start) * 1000
                    return RunResult(
                        output=output,
                        schema_valid=True,
                        retries=retries_from_attempts(attempt + 1),
                        provider_calls=provider_calls,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        latency_ms=elapsed,
                        raw_output=raw,
                        trace_events=trace_events,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    trace_events.append(
                        attempt_event(
                            attempt=attempt,
                            status="error",
                            detail=last_error,
                        )
                    )
                    prompt = (
                        f"{user_prompt}\n\nPrevious attempt failed validation: {exc}. "
                        "Return a value that matches the schema."
                    )
        except ImportError:
            last_error = "pydantic-ai package not installed; pip install -e '.[pydantic_ai]'"
        except Exception as e:
            last_error = str(e)

        elapsed = (time.perf_counter() - start) * 1000
        return RunResult(
            output=None,
            schema_valid=False,
            retries=retries_from_attempts(provider_calls) if provider_calls else 0,
            provider_calls=provider_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed,
            error=last_error,
            raw_output=raw,
            trace_events=trace_events,
        )

    async def run_extraction(self, document: str, subtype: str = "invoice") -> RunResult:
        schema_cls = InvoiceExtraction if subtype == "invoice" else ComplaintExtraction
        return await self._run_agent(
            schema_cls=schema_cls,
            system_prompt=get_extraction_prompt(subtype),
            user_prompt=document,
        )

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_agent(
            schema_cls=SQLGenerationResult,
            system_prompt=get_task_system_prompt("sql_generation"),
            user_prompt=build_task_user_prompt("sql_generation", sample),
        )

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_agent(
            schema_cls=ExecutiveSummary,
            system_prompt=get_task_system_prompt("summarization"),
            user_prompt=build_task_user_prompt("summarization", sample),
        )

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_agent(
            schema_cls=PolicyAnswer,
            system_prompt=get_task_system_prompt("compliance_qa"),
            user_prompt=build_task_user_prompt("compliance_qa", sample),
        )

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        result = await self._run_agent(
            schema_cls=WorkflowResult,
            system_prompt=get_task_system_prompt("multi_step"),
            user_prompt=build_task_user_prompt("multi_step", sample),
        )
        state_path = ["classify", "select_tool"]
        if result.output is not None:
            tool_name = getattr(result.output, "tool_used", None) or ""
            tool_result = execute_workflow_tool(str(tool_name), sample)
            state_path.extend(["execute_tool", "finalize"])
            result.tool_result = tool_result
            result.state_path = state_path
            result.trace_events = list(result.trace_events) + [
                {"type": "tool_execution", "tool_result": tool_result},
                {"type": "state_path", "state_path": state_path},
            ]
        else:
            result.state_path = state_path
        return result

    def score_extraction(self, predicted: Any, gold: dict[str, Any]) -> dict[str, float]:
        if predicted is None:
            return {k: 0.0 for k in gold}
        return field_accuracy(predicted.model_dump(mode="json"), gold)
