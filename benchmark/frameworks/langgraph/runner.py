"""LangGraph implementation for structured benchmark tasks."""

from __future__ import annotations

import os
import time
from typing import Any, TypedDict

from data.schemas.compliance import PolicyAnswer
from data.schemas.extraction import ComplaintExtraction, InvoiceExtraction
from data.schemas.sql import SQLGenerationResult
from data.schemas.summarization import ExecutiveSummary
from data.schemas.workflow import WorkflowResult
from eval.attempt_trace import retries_from_attempts
from eval.metrics import execute_workflow_tool, field_accuracy
from frameworks.base import FrameworkRunner, RunResult
from tasks.extraction.prompts import get_extraction_prompt
from tasks.prompts import build_task_user_prompt, get_task_system_prompt


class StructuredState(TypedDict):
    user_prompt: str
    system_prompt: str
    raw: str | None
    output_json: dict[str, Any] | None
    error: str | None
    retries: int
    tokens_in: int
    tokens_out: int


def _with_structured_output(llm: Any, schema_cls: type) -> Any:
    return llm.with_structured_output(
        schema_cls,
        include_raw=True,
        method="function_calling",
    )


def _build_structured_graph(schema_cls: type, model_name: str, max_retries: int):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    structured_llm = _with_structured_output(llm, schema_cls)

    def extract_node(state: StructuredState) -> dict[str, Any]:
        messages = [
            SystemMessage(content=state["system_prompt"]),
            HumanMessage(content=state["user_prompt"]),
        ]
        response = structured_llm.invoke(messages)
        parsed = response.get("parsed")
        raw_msg = response.get("raw")
        raw = raw_msg.content if raw_msg and hasattr(raw_msg, "content") else None
        usage = getattr(raw_msg, "usage_metadata", None) or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)

        if parsed is not None:
            return {
                "output_json": parsed.model_dump(mode="json"),
                "raw": raw or parsed.model_dump_json(),
                "error": None,
                "tokens_in": state["tokens_in"] + tokens_in,
                "tokens_out": state["tokens_out"] + tokens_out,
            }

        parsing_error = response.get("parsing_error")
        return {
            "raw": raw,
            "error": str(parsing_error) if parsing_error else "structured output parse failed",
            "retries": state["retries"] + 1,
            "tokens_in": state["tokens_in"] + tokens_in,
            "tokens_out": state["tokens_out"] + tokens_out,
        }

    def should_retry(state: StructuredState) -> str:
        if state.get("output_json") is not None:
            return "done"
        if state.get("retries", 0) >= max_retries:
            return "done"
        return "extract"

    graph = StateGraph(StructuredState)
    graph.add_node("extract", extract_node)
    graph.add_conditional_edges("extract", should_retry, {"extract": "extract", "done": END})
    graph.add_edge(START, "extract")
    return graph.compile()


class LangGraphRunner(FrameworkRunner):
    name = "langgraph"

    def __init__(self, model: str = "openai:gpt-4o-mini", max_retries: int = 3):
        self.model = model.replace("openai:", "")
        self.max_retries = max_retries
        self._graphs: dict[str, Any] = {}

    def _get_graph(self, schema_cls: type):
        key = schema_cls.__name__
        if key not in self._graphs:
            self._graphs[key] = _build_structured_graph(
                schema_cls, self.model, self.max_retries
            )
        return self._graphs[key]

    async def _run_graph(
        self,
        *,
        schema_cls: type,
        system_prompt: str,
        user_prompt: str,
    ) -> RunResult:
        start = time.perf_counter()
        raw: str | None = None

        try:
            graph = self._get_graph(schema_cls)
            final = await graph.ainvoke(
                {
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "raw": None,
                    "output_json": None,
                    "error": None,
                    "retries": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                }
            )
            elapsed = (time.perf_counter() - start) * 1000
            output_json = final.get("output_json")
            raw = final.get("raw")
            graph_retries = final.get("retries", 0)
            provider_calls = graph_retries + 1
            if output_json is not None:
                parsed = schema_cls.model_validate(output_json)
                return RunResult(
                    output=parsed,
                    schema_valid=True,
                    retries=retries_from_attempts(provider_calls),
                    provider_calls=provider_calls,
                    tokens_in=final.get("tokens_in", 0),
                    tokens_out=final.get("tokens_out", 0),
                    latency_ms=elapsed,
                    raw_output=raw,
                )
            return RunResult(
                output=None,
                schema_valid=False,
                retries=retries_from_attempts(provider_calls),
                provider_calls=provider_calls,
                tokens_in=final.get("tokens_in", 0),
                tokens_out=final.get("tokens_out", 0),
                latency_ms=elapsed,
                error=final.get("error"),
                raw_output=raw,
            )
        except ImportError:
            error = "langgraph not installed; pip install -e '.[langgraph]'"
        except Exception as e:
            error = str(e)
            elapsed = (time.perf_counter() - start) * 1000
            return RunResult(
                output=None,
                schema_valid=False,
                latency_ms=elapsed,
                error=error,
                raw_output=raw,
            )

        elapsed = (time.perf_counter() - start) * 1000
        return RunResult(
            output=None,
            schema_valid=False,
            latency_ms=elapsed,
            error=error,
            raw_output=raw,
        )

    async def run_extraction(self, document: str, subtype: str = "invoice") -> RunResult:
        schema_cls = InvoiceExtraction if subtype == "invoice" else ComplaintExtraction
        return await self._run_graph(
            schema_cls=schema_cls,
            system_prompt=get_extraction_prompt(subtype),
            user_prompt=document,
        )

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_graph(
            schema_cls=SQLGenerationResult,
            system_prompt=get_task_system_prompt("sql_generation"),
            user_prompt=build_task_user_prompt("sql_generation", sample),
        )

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_graph(
            schema_cls=ExecutiveSummary,
            system_prompt=get_task_system_prompt("summarization"),
            user_prompt=build_task_user_prompt("summarization", sample),
        )

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        return await self._run_graph(
            schema_cls=PolicyAnswer,
            system_prompt=get_task_system_prompt("compliance_qa"),
            user_prompt=build_task_user_prompt("compliance_qa", sample),
        )

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        result = await self._run_graph(
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
