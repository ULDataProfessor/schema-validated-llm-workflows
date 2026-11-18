"""Base protocol for framework runners."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    output: Any
    schema_valid: bool
    retries: int = 0
    provider_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    raw_output: str | None = None
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    field_scores: dict[str, float] = field(default_factory=dict)
    state_path: list[str] = field(default_factory=list)
    tool_result: dict[str, Any] | None = None


class FrameworkRunner(ABC):
    name: str

    @abstractmethod
    async def run_extraction(self, document: str, subtype: str = "invoice") -> RunResult:
        ...

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        raise NotImplementedError(f"{self.name} does not implement SQL generation")

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        raise NotImplementedError(f"{self.name} does not implement summarization")

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        raise NotImplementedError(f"{self.name} does not implement compliance QA")

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        raise NotImplementedError(f"{self.name} does not implement multi-step workflow")

    @staticmethod
    def _parsed_output(output: Any) -> Any:
        if output is None:
            return None
        if hasattr(output, "model_dump"):
            return output.model_dump(mode="json", exclude_none=True)
        if isinstance(output, dict):
            return output
        return None

    @staticmethod
    def _mean_score(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    @classmethod
    def _quality_score(
        cls,
        *,
        task: str,
        field_scores: dict[str, float],
        schema_valid: bool,
    ) -> tuple[float, str]:
        """Return the task-specific primary quality score used in paper tables."""
        if not field_scores:
            return (1.0 if schema_valid else 0.0), "schema_validity"

        if task == "sql_generation":
            return float(field_scores.get("result_correct", 0.0)), "result_correct"

        if task == "summarization":
            fact_scores = [
                float(value)
                for key, value in field_scores.items()
                if key.startswith("fact_")
                or key in {
                    "risk_coverage",
                    "recommendation_coverage",
                    "unsupported_content",
                }
            ]
            return cls._mean_score(fact_scores), "source_fact_coverage"

        if task == "compliance_qa":
            support_scores = [
                float(field_scores[key])
                for key in ("answer", "citation_support", "unsupported_claims")
                if key in field_scores
            ]
            return cls._mean_score(support_scores), "supported_answer_score"

        if task == "multi_step":
            workflow_scores = [
                float(field_scores[key])
                for key in (
                    "request_classification",
                    "tool_used",
                    "requires_human_review",
                    "tool_execution",
                    "state_path",
                )
                if key in field_scores
            ]
            return cls._mean_score(workflow_scores), "workflow_execution_score"

        return cls._mean_score([float(value) for value in field_scores.values()]), "field_macro_accuracy"

    def to_trial_record(
        self,
        *,
        task: str,
        sample_id: str,
        seed: int,
        result: RunResult,
        template_id: str | None = None,
        runner_schema_valid: bool | None = None,
        condition_order: int | None = None,
        ablation_id: str | None = None,
        ablation_condition: str | None = None,
        stress_id: str | None = None,
    ) -> dict[str, Any]:
        field_f1 = (
            sum(result.field_scores.values()) / len(result.field_scores)
            if result.field_scores
            else (1.0 if result.schema_valid else 0.0)
        )
        quality_score, quality_metric = self._quality_score(
            task=task,
            field_scores=result.field_scores,
            schema_valid=result.schema_valid,
        )
        provider_calls = result.provider_calls or (
            result.retries + 1 if result.schema_valid or result.retries else 0
        )
        record = {
            "trial_id": str(uuid.uuid4()),
            "framework": self.name,
            "task": task,
            "sample_id": sample_id,
            "template_id": template_id,
            "seed": seed,
            "replicate": seed,
            "schema_valid": result.schema_valid,
            "runner_schema_valid": (
                result.schema_valid if runner_schema_valid is None else runner_schema_valid
            ),
            "field_scores": result.field_scores,
            "field_f1": round(field_f1, 4),
            "quality_score": round(quality_score, 4),
            "quality_metric": quality_metric,
            "retries": result.retries,
            "provider_calls": provider_calls,
            "latency_ms": round(result.latency_ms, 1),
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "error": result.error,
            "raw_output": result.raw_output,
            "parsed_output": self._parsed_output(result.output),
            "trace_events": result.trace_events,
            "state_path": list(result.state_path),
            "tool_result": result.tool_result,
            "condition_order": condition_order,
        }
        if ablation_id is not None:
            record["ablation_id"] = ablation_id
            record["condition"] = ablation_condition
        if stress_id is not None:
            record["stress_id"] = stress_id
        return record


class DryRunRunner(FrameworkRunner):
    """Mock runner for pipeline testing without API calls."""

    name = "dry_run"

    @staticmethod
    def _parse_invoice_from_text(document: str) -> Any:
        import re
        from datetime import date
        from decimal import Decimal

        from data.schemas.extraction import InvoiceExtraction

        def field(pattern: str, default: str = "") -> str:
            m = re.search(pattern, document, re.IGNORECASE | re.MULTILINE)
            return m.group(1).strip() if m else default

        vendor = field(r"Vendor:\s*(.+)")
        inv_num = field(r"Invoice Number:\s*(.+)")
        inv_date = field(r"^Date:\s*(\d{4}-\d{2}-\d{2})", "") or "2024-01-01"
        due = field(r"Due Date:\s*(\d{4}-\d{2}-\d{2})") or None
        subtotal = field(r"Subtotal:\s*\$?([\d.]+)", "0")
        tax = field(r"Tax:\s*\$?([\d.]+)", "0")
        total = field(r"TOTAL DUE:\s*\$?([\d.]+)", subtotal)
        line_items = int(field(r"Line Items:\s*(\d+)", "1") or "1")
        return InvoiceExtraction(
            vendor_name=vendor or "Unknown Vendor",
            invoice_number=inv_num or "UNKNOWN",
            invoice_date=date.fromisoformat(inv_date),
            due_date=date.fromisoformat(due) if due else None,
            subtotal=Decimal(subtotal),
            tax_amount=Decimal(tax),
            total_amount=Decimal(total),
            line_item_count=line_items,
        )

    @staticmethod
    def _parse_complaint_from_text(document: str) -> Any:
        import re

        from data.schemas.extraction import ComplaintExtraction

        def field(pattern: str, default: str = "") -> str:
            m = re.search(pattern, document, re.IGNORECASE)
            return m.group(1).strip() if m else default

        return ComplaintExtraction(
            product_or_service=field(r"Product/Service:\s*(.+)") or "Unknown",
            issue_category=field(r"(\w+) problem", "other"),
            severity=field(r"Severity assessed as (\w+)", "medium"),
            requested_resolution=field(r"Requested resolution:\s*([^.]+)", "Unknown"),
            sentiment="negative",
        )

    async def run_extraction(self, document: str, subtype: str = "invoice") -> RunResult:
        start = time.perf_counter()
        if subtype == "invoice":
            output = self._parse_invoice_from_text(document)
        else:
            output = self._parse_complaint_from_text(document)
        elapsed = (time.perf_counter() - start) * 1000
        return RunResult(
            output=output,
            schema_valid=True,
            retries=0,
            tokens_in=0,
            tokens_out=0,
            latency_ms=elapsed,
            raw_output=output.model_dump_json(),
        )

    async def run_sql_generation(self, sample: dict[str, Any]) -> RunResult:
        from data.schemas.sql import SQLGenerationResult

        start = time.perf_counter()
        output = SQLGenerationResult.model_validate(sample["gold"])
        elapsed = (time.perf_counter() - start) * 1000
        return RunResult(
            output=output,
            schema_valid=True,
            latency_ms=elapsed,
            raw_output=output.model_dump_json(),
        )

    async def run_summarization(self, sample: dict[str, Any]) -> RunResult:
        from data.schemas.summarization import ExecutiveSummary

        start = time.perf_counter()
        output = ExecutiveSummary.model_validate(sample["gold"])
        elapsed = (time.perf_counter() - start) * 1000
        return RunResult(
            output=output,
            schema_valid=True,
            latency_ms=elapsed,
            raw_output=output.model_dump_json(),
        )

    async def run_compliance_qa(self, sample: dict[str, Any]) -> RunResult:
        from data.schemas.compliance import PolicyAnswer

        start = time.perf_counter()
        output = PolicyAnswer.model_validate(sample["gold"])
        elapsed = (time.perf_counter() - start) * 1000
        return RunResult(
            output=output,
            schema_valid=True,
            latency_ms=elapsed,
            raw_output=output.model_dump_json(),
        )

    async def run_multi_step(self, sample: dict[str, Any]) -> RunResult:
        from data.schemas.workflow import WorkflowResult
        from eval.metrics import execute_workflow_tool

        start = time.perf_counter()
        gold = {
            key: value
            for key, value in sample["gold"].items()
            if key not in {"expected_tool_result", "expected_state_path"}
        }
        output = WorkflowResult.model_validate(gold)
        tool_result = execute_workflow_tool(output.tool_used, sample)
        state_path = ["classify", "select_tool", "execute_tool", "finalize"]
        elapsed = (time.perf_counter() - start) * 1000
        return RunResult(
            output=output,
            schema_valid=True,
            provider_calls=1,
            latency_ms=elapsed,
            raw_output=output.model_dump_json(),
            tool_result=tool_result,
            state_path=state_path,
            trace_events=[
                {"type": "tool_execution", "tool_result": tool_result},
                {"type": "state_path", "state_path": state_path},
            ],
        )
