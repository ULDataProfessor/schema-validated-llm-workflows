"""CrewAI implementation for structured benchmark tasks."""

from __future__ import annotations

import json
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


def _crewai_model_name(model: str) -> str:
    """Convert benchmark model names into CrewAI/LiteLLM-style model names."""
    if model.startswith("openai:"):
        return model.replace("openai:", "openai/", 1)
    return model


def _usage_from_crew_output(output: Any, crew: Any | None = None) -> tuple[int, int]:
    """Best-effort token extraction across CrewAI output versions."""

    def visit(value: Any) -> tuple[int, int]:
        if value is None:
            return 0, 0
        if isinstance(value, dict):
            tokens_in = (
                value.get("input_tokens")
                or value.get("prompt_tokens")
                or value.get("total_prompt_tokens")
                or 0
            )
            tokens_out = (
                value.get("output_tokens")
                or value.get("completion_tokens")
                or value.get("total_completion_tokens")
                or 0
            )
            if tokens_in or tokens_out:
                return int(tokens_in or 0), int(tokens_out or 0)
            total_in = total_out = 0
            for item in value.values():
                item_in, item_out = visit(item)
                total_in += item_in
                total_out += item_out
            return total_in, total_out
        if isinstance(value, (list, tuple)):
            total_in = total_out = 0
            for item in value:
                item_in, item_out = visit(item)
                total_in += item_in
                total_out += item_out
            return total_in, total_out
        tokens_in = (
            getattr(value, "input_tokens", None)
            or getattr(value, "prompt_tokens", None)
            or getattr(value, "total_prompt_tokens", None)
            or 0
        )
        tokens_out = (
            getattr(value, "output_tokens", None)
            or getattr(value, "completion_tokens", None)
            or getattr(value, "total_completion_tokens", None)
            or 0
        )
        return int(tokens_in or 0), int(tokens_out or 0)

    for source in (
        getattr(output, "token_usage", None),
        getattr(output, "usage_metrics", None),
        getattr(crew, "usage_metrics", None),
    ):
        tokens_in, tokens_out = visit(source)
        if tokens_in or tokens_out:
            return tokens_in, tokens_out
    return 0, 0


def _raw_crew_output(output: Any, task: Any | None = None) -> str | None:
    for value in (
        getattr(output, "raw", None),
        getattr(getattr(task, "output", None), "raw", None),
    ):
        if value:
            return str(value)
    if hasattr(output, "to_dict"):
        try:
            return json.dumps(output.to_dict())
        except TypeError:
            return str(output.to_dict())
    return str(output) if output is not None else None


def _structured_crew_value(output: Any, task: Any | None = None) -> Any:
    for value in (
        getattr(output, "pydantic", None),
        getattr(getattr(task, "output", None), "pydantic", None),
        getattr(output, "json_dict", None),
        getattr(getattr(task, "output", None), "json_dict", None),
    ):
        if value is not None:
            return value
    tasks_output = getattr(output, "tasks_output", None) or []
    for task_output in reversed(tasks_output):
        value = getattr(task_output, "pydantic", None) or getattr(task_output, "json_dict", None)
        if value is not None:
            return value
    return None


def _coerce_schema_output(value: Any, schema_cls: type) -> Any:
    if isinstance(value, schema_cls):
        return value
    if isinstance(value, str):
        return schema_cls.model_validate_json(value)
    return schema_cls.model_validate(value)


class CrewAIRunner(FrameworkRunner):
    name = "crewai"

    def __init__(self, model: str = "openai:gpt-4o-mini", max_retries: int = 3):
        self.model = _crewai_model_name(model)
        self.max_retries = max_retries

    async def _run_crew(
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
            from crewai import Agent, Crew, LLM, Process, Task
        except ImportError:
            elapsed = (time.perf_counter() - start) * 1000
            return RunResult(
                output=None,
                schema_valid=False,
                latency_ms=elapsed,
                error="crewai package not installed; pip install -e '.[crewai]'",
            )

        prompt = user_prompt
        for attempt in range(self.max_retries + 1):
            provider_calls += 1
            try:
                llm = LLM(model=self.model, temperature=0.0)
                agent = Agent(
                    role="Structured business workflow agent",
                    goal="Return validated structured data for the assigned business task.",
                    backstory=(
                        "You produce concise, auditable outputs and follow the provided "
                        "schema exactly."
                    ),
                    llm=llm,
                    allow_delegation=False,
                    max_retry_limit=self.max_retries,
                    verbose=False,
                )
                task = Task(
                    description=f"{system_prompt}\n\nInput:\n{prompt}",
                    expected_output=(
                        f"A valid {schema_cls.__name__} object using only supported facts "
                        "from the input."
                    ),
                    agent=agent,
                    output_pydantic=schema_cls,
                )
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=False,
                )
                if hasattr(crew, "kickoff_async"):
                    crew_output = await crew.kickoff_async()
                else:
                    import asyncio

                    crew_output = await asyncio.to_thread(crew.kickoff)
                raw = _raw_crew_output(crew_output, task)
                value = _structured_crew_value(crew_output, task)
                output = _coerce_schema_output(value, schema_cls)
                usage_in, usage_out = _usage_from_crew_output(crew_output, crew)
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
                    raw_output=raw or output.model_dump_json(),
                    trace_events=trace_events,
                )
            except Exception as exc:
                last_error = str(exc)
                trace_events.append(
                    attempt_event(
                        attempt=attempt,
                        status="validation_error",
                        detail=str(exc),
                    )
                )
                prompt = (
                    f"{user_prompt}\n\n"
                    f"Previous structured-output validation error: {exc}\n"
                    f"Return a value that satisfies the {schema_cls.__name__} schema."
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
        return await self._run_crew(
            schema_cls=schema_cls,
            system_prompt=get_extraction_prompt(subtype),
            user_prompt=document,
        )

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_crew(
            schema_cls=SQLGenerationResult,
            system_prompt=get_task_system_prompt("sql_generation"),
            user_prompt=build_task_user_prompt("sql_generation", sample),
        )

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_crew(
            schema_cls=ExecutiveSummary,
            system_prompt=get_task_system_prompt("summarization"),
            user_prompt=build_task_user_prompt("summarization", sample),
        )

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_crew(
            schema_cls=PolicyAnswer,
            system_prompt=get_task_system_prompt("compliance_qa"),
            user_prompt=build_task_user_prompt("compliance_qa", sample),
        )

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        result = await self._run_crew(
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
