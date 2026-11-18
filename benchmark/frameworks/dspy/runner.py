"""DSPy implementation for structured benchmark tasks."""

from __future__ import annotations

import json
import os
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


def _dspy_model_name(model: str) -> str:
    """Convert benchmark model names into LiteLLM-style DSPy model names."""
    if model.startswith("openai:"):
        return model.replace("openai:", "openai/", 1)
    return model


def _usage_from_prediction(prediction: Any) -> tuple[int, int]:
    """Best-effort token extraction across DSPy prediction/LM usage formats."""

    def visit(value: Any) -> tuple[int, int]:
        if value is None:
            return 0, 0
        if isinstance(value, dict):
            tokens_in = (
                value.get("input_tokens")
                or value.get("prompt_tokens")
                or value.get("request_tokens")
                or 0
            )
            tokens_out = (
                value.get("output_tokens")
                or value.get("completion_tokens")
                or value.get("response_tokens")
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

    usage = None
    if hasattr(prediction, "get_lm_usage"):
        usage = prediction.get_lm_usage()
    if usage is None:
        usage = getattr(prediction, "_lm_usage", None)
    return visit(usage)


def _make_signature(dspy: Any, schema_cls: type, system_prompt: str) -> type:
    """Create a DSPy signature with one prompt input and one typed output."""
    return type(
        f"{schema_cls.__name__}DSPySignature",
        (dspy.Signature,),
        {
            "__doc__": system_prompt,
            "__annotations__": {
                "task_input": str,
                "output": schema_cls,
            },
            "task_input": dspy.InputField(desc="Task input, including source text and context."),
            "output": dspy.OutputField(desc=f"Validated {schema_cls.__name__} object."),
        },
    )


def _prediction_value(prediction: Any) -> Any:
    if hasattr(prediction, "get"):
        return prediction.get("output")
    return getattr(prediction, "output", None)


def _raw_prediction(prediction: Any, value: Any) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if hasattr(prediction, "toDict"):
        try:
            return json.dumps(prediction.toDict())
        except TypeError:
            pass
    return str(value)


def _coerce_schema_output(value: Any, schema_cls: type) -> Any:
    if isinstance(value, schema_cls):
        return value
    if isinstance(value, str):
        return schema_cls.model_validate_json(value)
    return schema_cls.model_validate(value)


class DSPyRunner(FrameworkRunner):
    name = "dspy"

    def __init__(self, model: str = "openai:gpt-4o-mini", max_retries: int = 3):
        self.model = _dspy_model_name(model)
        self.max_retries = max_retries

    async def _run_predict(
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
            import dspy
        except ImportError:
            elapsed = (time.perf_counter() - start) * 1000
            return RunResult(
                output=None,
                schema_valid=False,
                latency_ms=elapsed,
                error="dspy package not installed; pip install -e '.[dspy]'",
            )

        signature = _make_signature(dspy, schema_cls, system_prompt)
        predictor = dspy.Predict(signature)
        lm = dspy.LM(
            self.model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.0,
        )
        prompt = user_prompt

        for attempt in range(self.max_retries + 1):
            provider_calls += 1
            try:
                with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
                    prediction = predictor(task_input=prompt)
                value = _prediction_value(prediction)
                raw = _raw_prediction(prediction, value)
                output = _coerce_schema_output(value, schema_cls)
                usage_in, usage_out = _usage_from_prediction(prediction)
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
        return await self._run_predict(
            schema_cls=schema_cls,
            system_prompt=get_extraction_prompt(subtype),
            user_prompt=document,
        )

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_predict(
            schema_cls=SQLGenerationResult,
            system_prompt=get_task_system_prompt("sql_generation"),
            user_prompt=build_task_user_prompt("sql_generation", sample),
        )

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_predict(
            schema_cls=ExecutiveSummary,
            system_prompt=get_task_system_prompt("summarization"),
            user_prompt=build_task_user_prompt("summarization", sample),
        )

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_predict(
            schema_cls=PolicyAnswer,
            system_prompt=get_task_system_prompt("compliance_qa"),
            user_prompt=build_task_user_prompt("compliance_qa", sample),
        )

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        result = await self._run_predict(
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
