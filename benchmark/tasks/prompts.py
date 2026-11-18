"""Shared prompts for benchmark tasks beyond document extraction."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPTS = {
    "sql_generation": (
        "You are a business analytics SQL assistant. Generate a read-only SQLite "
        "query that answers the user's question using only the provided schema. "
        "Return a structured object with sql, explanation, and tables_used."
    ),
    "summarization": (
        "You are an executive reporting assistant. Summarize the report into the "
        "required structured sections. Preserve numeric facts from the source and "
        "do not add unsupported claims."
    ),
    "compliance_qa": (
        "You are a compliance question-answering assistant. Answer only from the "
        "policy document. Include citations with verbatim quotes from the document "
        "and list unsupported claims when the answer cannot be fully supported."
    ),
    "multi_step": (
        "You are a business workflow routing assistant. Classify the request, select "
        "the most appropriate tool, extract relevant entities, and produce a final "
        "structured recommendation."
    ),
}


def get_task_system_prompt(task: str) -> str:
    try:
        return SYSTEM_PROMPTS[task]
    except KeyError as exc:
        raise ValueError(f"No system prompt configured for task: {task}") from exc


def build_task_user_prompt(task: str, sample: dict[str, Any]) -> str:
    if task == "sql_generation":
        return (
            "Database schema:\n"
            f"{sample['schema_ddl']}\n\n"
            "Business question:\n"
            f"{sample['question']}\n\n"
            "Return a read-only SELECT query."
        )
    if task == "summarization":
        return f"Business report:\n{sample['text']}"
    if task == "compliance_qa":
        return (
            "Policy document:\n"
            f"{sample['document']}\n\n"
            "Question:\n"
            f"{sample['question']}"
        )
    if task == "multi_step":
        return (
            "Allowed classifications: billing, technical, refund, escalation, general.\n"
            "Allowed tools: lookup_invoice, open_support_ticket, create_refund_case, "
            "notify_manager, send_help_article.\n\n"
            "Request:\n"
            f"{sample['text']}"
        )
    raise ValueError(f"No user prompt builder configured for task: {task}")
