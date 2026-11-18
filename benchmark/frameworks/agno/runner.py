"""Agno implementation for structured benchmark tasks."""

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


def _usage_from_response(response: Any) -> tuple[int, int]:
    """Best-effort token extraction across Agno response versions."""
    for attr in ("metrics", "usage"):
        value = getattr(response, attr, None)
        if value is None:
            continue
        if isinstance(value, dict):
            tokens_in = value.get("input_tokens") or value.get("prompt_tokens") or 0
            tokens_out = value.get("output_tokens") or value.get("completion_tokens") or 0
            return int(tokens_in or 0), int(tokens_out or 0)
        tokens_in = (
            getattr(value, "input_tokens", None)
            or getattr(value, "prompt_tokens", None)
            or getattr(value, "request_tokens", None)
            or 0
        )
        tokens_out = (
            getattr(value, "output_tokens", None)
            or getattr(value, "completion_tokens", None)
            or getattr(value, "response_tokens", None)
            or 0
        )
        return int(tokens_in or 0), int(tokens_out or 0)
    return 0, 0


class AgnoRunner(FrameworkRunner):
    name = "agno"

    def __init__(self, model: str = "openai:gpt-4o-mini", max_retries: int = 3):
        self.model = model.replace("openai:", "")
        self.max_retries = max_retries

    async def _run_agent(
        self,
        *,
        schema_cls: type,
        system_prompt: str,
        user_prompt: str,
    ) -> RunResult:
        start = time.perf_counter()
        raw: str | None = None
        last_error: str | None = None
        tokens_in = tokens_out = 0
        provider_calls = 0
        trace_events: list[dict[str, Any]] = []

        try:
            from agno.agent import Agent
            from agno.models.openai import OpenAIResponses
        except ImportError:
            elapsed = (time.perf_counter() - start) * 1000
            return RunResult(
                output=None,
                schema_valid=False,
                latency_ms=elapsed,
                error="agno package not installed; pip install -e '.[agno]'",
            )

        for attempt in range(self.max_retries + 1):
            provider_calls += 1
            try:
                agent = Agent(
                    model=OpenAIResponses(id=self.model),
                    instructions=system_prompt,
                    output_schema=schema_cls,
                )
                response = agent.run(user_prompt)
                content = response.content
                output = (
                    content
                    if isinstance(content, schema_cls)
                    else schema_cls.model_validate(content)
                )
                raw = output.model_dump_json()
                usage_in, usage_out = _usage_from_response(response)
                tokens_in += usage_in
                tokens_out += usage_out
                trace_events.append(attempt_event(attempt=attempt, status="schema_valid"))
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
            except Exception as exc:
                last_error = str(exc)
                trace_events.append(
                    attempt_event(attempt=attempt, status="error", detail=last_error)
                )

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
