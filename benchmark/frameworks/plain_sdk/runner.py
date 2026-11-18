"""Plain OpenAI SDK baseline for structured benchmark tasks."""

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


def _strict_openai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Add additionalProperties=false required by OpenAI strict JSON schema mode."""

    def patch(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
            for value in node.values():
                patch(value)
        elif isinstance(node, list):
            for item in node:
                patch(item)

    patched = json.loads(json.dumps(schema))
    patch(patched)
    return patched


class PlainSDKRunner(FrameworkRunner):
    name = "plain_sdk"

    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 3):
        self.model = model.replace("openai:", "")
        self.max_retries = max_retries

    async def _run_structured(
        self,
        *,
        schema_cls: type,
        system_prompt: str,
        user_prompt: str,
    ) -> RunResult:
        start = time.perf_counter()
        last_error: str | None = None
        raw: str | None = None
        tokens_in = tokens_out = 0
        provider_calls = 0
        trace_events: list[dict] = []

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            for attempt in range(self.max_retries + 1):
                provider_calls += 1
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_cls.__name__,
                            "schema": _strict_openai_schema(schema_cls.model_json_schema()),
                            "strict": False,
                        },
                    },
                )
                usage = response.usage
                if usage:
                    tokens_in += usage.prompt_tokens or 0
                    tokens_out += usage.completion_tokens or 0
                raw = response.choices[0].message.content or ""
                try:
                    parsed = schema_cls.model_validate_json(raw)
                    trace_events.append(attempt_event(attempt=attempt, status="schema_valid"))
                    elapsed = (time.perf_counter() - start) * 1000
                    return RunResult(
                        output=parsed,
                        schema_valid=True,
                        retries=retries_from_attempts(attempt + 1),
                        provider_calls=provider_calls,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        latency_ms=elapsed,
                        raw_output=raw,
                        trace_events=trace_events,
                    )
                except Exception as e:
                    last_error = str(e)
                    trace_events.append(
                        attempt_event(attempt=attempt, status="validation_error", detail=str(e))
                    )
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Validation failed: {e}. Return valid JSON matching the schema.",
                        }
                    )
        except ImportError:
            last_error = "openai package not installed"
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
        return await self._run_structured(
            schema_cls=schema_cls,
            system_prompt=get_extraction_prompt(subtype),
            user_prompt=document,
        )

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_structured(
            schema_cls=SQLGenerationResult,
            system_prompt=get_task_system_prompt("sql_generation"),
            user_prompt=build_task_user_prompt("sql_generation", sample),
        )

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_structured(
            schema_cls=ExecutiveSummary,
            system_prompt=get_task_system_prompt("summarization"),
            user_prompt=build_task_user_prompt("summarization", sample),
        )

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_structured(
            schema_cls=PolicyAnswer,
            system_prompt=get_task_system_prompt("compliance_qa"),
            user_prompt=build_task_user_prompt("compliance_qa", sample),
        )

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        result = await self._run_structured(
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

    def score_extraction(
        self, predicted: Any, gold: dict[str, Any]
    ) -> dict[str, float]:
        if predicted is None:
            return {k: 0.0 for k in gold}
        return field_accuracy(predicted.model_dump(mode="json"), gold)
