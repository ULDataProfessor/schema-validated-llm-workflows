"""PydanticAI implementation for structured benchmark tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from data.schemas.compliance import PolicyAnswer
from data.schemas.extraction import ComplaintExtraction, InvoiceExtraction
from data.schemas.sql import SQLGenerationResult
from data.schemas.summarization import ExecutiveSummary
from data.schemas.workflow import WorkflowResult
from eval.attempt_trace import attempt_event, retries_from_attempts
from eval.metrics import execute_workflow_tool, field_accuracy, normalize_string
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


@dataclass
class TaskDeps:
    """Typed dependency bag for feature-using ablations."""

    task: str
    sample_id: str | None = None
    policy_text: str | None = None
    schema_context: str | None = None
    request_text: str | None = None


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

    def _register_validators(self, agent: Any, schema_cls: type) -> None:
        if not self.enable_validators:
            return
        from pydantic_ai import ModelRetry

        if schema_cls is SQLGenerationResult:

            @agent.output_validator
            async def validate_sql(_ctx: Any, output: SQLGenerationResult) -> SQLGenerationResult:
                sql = (output.sql or "").strip().lower()
                if not sql:
                    raise ModelRetry("SQL must be non-empty.")
                if any(tok in sql for tok in (" drop ", " delete ", " update ", " insert ", " alter ")):
                    raise ModelRetry("Only read-only SELECT-style SQL is allowed.")
                if "select" not in sql:
                    raise ModelRetry("SQL must include a SELECT clause.")
                return output

        elif schema_cls is PolicyAnswer:

            @agent.output_validator
            async def validate_policy(ctx: Any, output: PolicyAnswer) -> PolicyAnswer:
                if not output.citations:
                    raise ModelRetry("Provide at least one citation with a quote span.")
                quote = normalize_string(getattr(output.citations[0], "quote", "") or "")
                if len(quote) < 8:
                    raise ModelRetry("Citation quote is too short to be grounding evidence.")
                deps = getattr(ctx, "deps", None)
                policy = normalize_string(getattr(deps, "policy_text", None) or "")
                if policy and quote not in policy:
                    raise ModelRetry(
                        "Citation quote must appear in the provided policy text."
                    )
                return output

        elif schema_cls is WorkflowResult:

            @agent.output_validator
            async def validate_workflow(_ctx: Any, output: WorkflowResult) -> WorkflowResult:
                tool = (getattr(output, "tool_used", None) or "").strip()
                if not tool:
                    raise ModelRetry("Select a concrete tool_used value.")
                return output

    async def _run_agent(
        self,
        *,
        schema_cls: type,
        system_prompt: str,
        user_prompt: str,
        deps: TaskDeps | None = None,
    ) -> RunResult:
        start = time.perf_counter()
        raw: str | None = None
        tokens_in = tokens_out = 0
        trace_events: list[dict[str, Any]] = []
        last_error: str | None = None
        provider_calls = 0

        try:
            from pydantic_ai import Agent

            agent_kwargs: dict[str, Any] = {
                "model": self.model,
                "output_type": schema_cls,
                "system_prompt": system_prompt,
                "retries": 1,
            }
            if self.use_typed_deps:
                agent_kwargs["deps_type"] = TaskDeps
            agent = Agent(**agent_kwargs)
            self._register_validators(agent, schema_cls)

            prompt = user_prompt
            if self.use_typed_deps and deps is not None:
                prompt = (
                    f"{user_prompt}\n\n"
                    f"[typed_deps.task={deps.task}; sample_id={deps.sample_id}]"
                )

            for attempt in range(self.max_retries + 1):
                provider_calls += 1
                try:
                    if self.use_typed_deps and deps is not None:
                        result = await agent.run(prompt, deps=deps)
                    else:
                        result = await agent.run(prompt)
                    output = result.output
                    raw = output.model_dump_json()
                    call_in, call_out = _usage_from_result(result)
                    tokens_in += call_in or 0
                    tokens_out += call_out or 0
                    trace_events.append(
                        attempt_event(
                            attempt=attempt,
                            status="schema_valid",
                            detail=(
                                "validators_enabled"
                                if self.enable_validators
                                else "validators_disabled"
                            ),
                        )
                    )
                    if self.use_typed_deps:
                        trace_events.append(
                            {
                                "type": "typed_deps",
                                "task": getattr(deps, "task", None),
                                "sample_id": getattr(deps, "sample_id", None),
                            }
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
                        "Return a value that matches the schema and domain constraints."
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

    def _deps_for_sample(self, task: str, sample: dict[str, Any] | None) -> TaskDeps | None:
        if not self.use_typed_deps:
            return None
        sample = sample or {}
        return TaskDeps(
            task=task,
            sample_id=sample.get("id"),
            policy_text=sample.get("policy_text") or sample.get("policy"),
            schema_context=sample.get("schema") or sample.get("db_schema"),
            request_text=sample.get("text") or sample.get("request"),
        )

    async def run_extraction(self, document: str, subtype: str = "invoice") -> RunResult:
        schema_cls = InvoiceExtraction if subtype == "invoice" else ComplaintExtraction
        return await self._run_agent(
            schema_cls=schema_cls,
            system_prompt=get_extraction_prompt(subtype),
            user_prompt=document,
            deps=TaskDeps(task="extraction") if self.use_typed_deps else None,
        )

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_agent(
            schema_cls=SQLGenerationResult,
            system_prompt=get_task_system_prompt("sql_generation"),
            user_prompt=build_task_user_prompt("sql_generation", sample),
            deps=self._deps_for_sample("sql_generation", sample),
        )

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_agent(
            schema_cls=ExecutiveSummary,
            system_prompt=get_task_system_prompt("summarization"),
            user_prompt=build_task_user_prompt("summarization", sample),
            deps=self._deps_for_sample("summarization", sample),
        )

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_agent(
            schema_cls=PolicyAnswer,
            system_prompt=get_task_system_prompt("compliance_qa"),
            user_prompt=build_task_user_prompt("compliance_qa", sample),
            deps=self._deps_for_sample("compliance_qa", sample),
        )

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        result = await self._run_agent(
            schema_cls=WorkflowResult,
            system_prompt=get_task_system_prompt("multi_step"),
            user_prompt=build_task_user_prompt("multi_step", sample),
            deps=self._deps_for_sample("multi_step", sample),
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
