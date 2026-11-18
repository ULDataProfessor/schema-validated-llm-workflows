"""Independent schema revalidation at the benchmark harness boundary."""

from __future__ import annotations

from typing import Any

from data.schemas.compliance import PolicyAnswer
from data.schemas.extraction import ComplaintExtraction, InvoiceExtraction
from data.schemas.sql import SQLGenerationResult
from data.schemas.summarization import ExecutiveSummary
from data.schemas.workflow import WorkflowResult


TASK_SCHEMAS: dict[str, type] = {
    "sql_generation": SQLGenerationResult,
    "summarization": ExecutiveSummary,
    "compliance_qa": PolicyAnswer,
    "multi_step": WorkflowResult,
}


def schema_for_task(task: str, sample: dict[str, Any] | None = None) -> type:
    if task == "extraction":
        subtype = (sample or {}).get("subtype", "invoice")
        return InvoiceExtraction if subtype == "invoice" else ComplaintExtraction
    if task not in TASK_SCHEMAS:
        raise ValueError(f"No schema registered for task: {task}")
    return TASK_SCHEMAS[task]


def revalidate_output(
    task: str,
    output: Any,
    *,
    sample: dict[str, Any] | None = None,
) -> tuple[bool, Any | None, str | None]:
    """Validate runner output against the task schema.

    Returns (schema_valid, parsed_model_or_none, error_or_none).
    """
    schema_cls = schema_for_task(task, sample)
    if output is None:
        return False, None, "missing_output"
    try:
        if hasattr(output, "model_dump"):
            payload = output.model_dump(mode="json")
        elif isinstance(output, dict):
            payload = output
        elif isinstance(output, str):
            parsed = schema_cls.model_validate_json(output)
            return True, parsed, None
        else:
            return False, None, f"unsupported_output_type:{type(output).__name__}"
        parsed = schema_cls.model_validate(payload)
        return True, parsed, None
    except Exception as exc:  # noqa: BLE001 - boundary must fail closed
        return False, None, str(exc)
