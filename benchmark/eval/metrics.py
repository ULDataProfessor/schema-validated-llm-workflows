"""Evaluation metrics for benchmark trials."""

from __future__ import annotations

import re
import sqlite3
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "must",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
}


def normalize_string(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _canonical_value(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return normalize_string(str(value))


def _text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_text_content(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_text_content(v) for v in value)
    return str(value)


def _money_millions_present(text: str, amount: str) -> bool:
    normalized = normalize_string(text)
    escaped = re.escape(amount)
    return bool(
        re.search(rf"\$?\s*{escaped}\s*(?:m|million|millions)?\b", normalized)
    )


def _percent_present(text: str, percent: str) -> bool:
    normalized = normalize_string(text)
    escaped = re.escape(percent)
    return bool(re.search(rf"(?<![a-z0-9]){escaped}\s*%", normalized))


def phrase_coverage(predicted: Any, expected_terms: list[str]) -> float:
    """Fraction of expected terms or phrases appearing in predicted text."""
    if not expected_terms:
        return 1.0
    text = normalize_string(_text_content(predicted))
    matched = sum(1 for term in expected_terms if normalize_string(term) in text)
    return matched / len(expected_terms)


def summarization_fact_scores(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    """Deterministic coverage score for synthetic report summaries.

    The synthetic reports encode a small set of known facts. Exact wording is not
    required, but the generated structured summary should preserve the quarter,
    revenue amount, growth rate, named risk, and recommendation.
    """
    text = _text_content(predicted)
    headline = normalize_string(gold.get("headline", ""))
    revenue_summary = normalize_string(gold.get("revenue_summary", ""))
    quarter_match = re.search(r"\b(q[1-4])\b", headline)
    revenue_match = re.search(r"\$(\d+)m", revenue_summary)
    growth_match = re.search(r"(-?\d+)%", revenue_summary)

    return {
        "fact_quarter": float(
            bool(quarter_match and quarter_match.group(1) in normalize_string(text))
        ),
        "fact_revenue": float(
            bool(revenue_match and _money_millions_present(text, revenue_match.group(1)))
        ),
        "fact_growth": float(
            bool(growth_match and _percent_present(text, growth_match.group(1)))
        ),
        "risk_coverage": phrase_coverage(predicted, list(gold.get("key_risks", []))),
        "recommendation_coverage": phrase_coverage(
            predicted,
            list(gold.get("strategic_recommendations", [])),
        ),
    }


def token_f1(predicted: str, gold: str) -> float:
    """Token-set F1 for short answer paraphrase tolerance."""
    pred_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_string(predicted))
        if token not in STOPWORDS
    }
    gold_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_string(gold))
        if token not in STOPWORDS
    }
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def field_accuracy(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    """Per-field exact match after normalization (1.0 or 0.0)."""
    scores: dict[str, float] = {}
    for key, gold_val in gold.items():
        pred_val = predicted.get(key)
        if isinstance(gold_val, (int, float)) and isinstance(pred_val, (int, float)):
            scores[key] = 1.0 if float(pred_val) == float(gold_val) else 0.0
        else:
            scores[key] = 1.0 if _canonical_value(pred_val) == _canonical_value(gold_val) else 0.0
    return scores


def macro_f1(field_scores: list[dict[str, float]]) -> float:
    if not field_scores:
        return 0.0
    per_field: dict[str, list[float]] = {}
    for scores in field_scores:
        for k, v in scores.items():
            per_field.setdefault(k, []).append(v)
    if not per_field:
        return 0.0
    return sum(sum(v) / len(v) for v in per_field.values()) / len(per_field)


def citation_valid(quote: str, document: str) -> bool:
    """Check whether citation quote appears in source document (fuzzy whitespace)."""
    norm_doc = normalize_string(document)
    norm_quote = normalize_string(quote)
    return norm_quote in norm_doc if norm_quote else False


def citation_support_score(citations: list[dict[str, Any]], document: str) -> float:
    """Fraction of citations whose quoted text appears in the source document."""
    if not citations:
        return 0.0
    valid = sum(1 for citation in citations if citation_valid(citation.get("quote", ""), document))
    return valid / len(citations)


def required_fields_present(record: dict[str, Any], fields: list[str]) -> dict[str, float]:
    """Score whether required fields are present and non-empty."""
    scores: dict[str, float] = {}
    for field in fields:
        value = record.get(field)
        if value is None or value == "" or value == [] or value == {}:
            scores[field] = 0.0
        else:
            scores[field] = 1.0
    return scores


_WRITE_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|INTO|TRUNCATE)\b",
    re.IGNORECASE,
)


def sql_is_read_only(sql: str) -> bool:
    """Reject statements that are not a single read-only SELECT/WITH query."""
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return False
    if ";" in stripped:
        return False
    if _WRITE_SQL_RE.search(stripped):
        return False
    return bool(re.match(r"(?is)^\s*(WITH\b|SELECT\b)", stripped))


def sql_syntax_valid(sql: str, conn: sqlite3.Connection | None = None) -> bool:
    if not sql_is_read_only(sql):
        return False
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(":memory:")
    try:
        conn.execute(f"EXPLAIN {sql}")
        return True
    except sqlite3.Error:
        return False
    finally:
        if own_conn and conn:
            conn.close()


def sql_execution_success(sql: str, conn: sqlite3.Connection) -> tuple[bool, list[tuple] | None]:
    if not sql_is_read_only(sql):
        return False, None
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return True, rows
    except sqlite3.Error:
        return False, None


def result_sets_equal(a: list[tuple] | None, b: list[tuple] | None) -> bool:
    if a is None or b is None:
        return False
    return sorted(a) == sorted(b)


def connect_sample_database(setup_sql: str) -> sqlite3.Connection:
    """Create an in-memory SQLite database from benchmark setup SQL."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(setup_sql)
    return conn


def score_sql_result(predicted_sql: str, gold_sql: str, setup_sql: str) -> dict[str, float]:
    """Score SQL on separate fresh DBs with read-only enforcement."""
    pred_conn = connect_sample_database(setup_sql)
    gold_conn = connect_sample_database(setup_sql)
    try:
        read_only = float(sql_is_read_only(predicted_sql))
        syntax_valid = float(sql_syntax_valid(predicted_sql, pred_conn))
        executed, rows = sql_execution_success(predicted_sql, pred_conn)
        gold_executed, gold_rows = sql_execution_success(gold_sql, gold_conn)
        result_correct = executed and gold_executed and result_sets_equal(rows, gold_rows)
        return {
            "read_only": read_only,
            "syntax_valid": syntax_valid,
            "execution_success": float(executed),
            "result_correct": float(result_correct),
        }
    finally:
        pred_conn.close()
        gold_conn.close()


def summarization_unsupported_penalty(predicted: dict[str, Any], gold: dict[str, Any]) -> float:
    """1.0 if no forbidden unsupported terms appear; else 0.0."""
    forbidden = list(gold.get("unsupported_forbidden_terms", []))
    if not forbidden:
        return 1.0
    text = normalize_string(_text_content(predicted))
    for term in forbidden:
        if normalize_string(term) in text:
            return 0.0
    return 1.0


def summarization_quality_scores(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    scores = summarization_fact_scores(predicted, gold)
    scores["unsupported_content"] = summarization_unsupported_penalty(predicted, gold)
    return scores


def compliance_claim_grounding_scores(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    document: str,
) -> dict[str, float]:
    """Answer support plus citation grounding and unsupported-claim penalty."""
    answer_score = token_f1(str(predicted.get("answer", "")), str(gold.get("answer", "")))
    citation_score = citation_support_score(predicted.get("citations", []), document)
    predicted_unsupported = predicted.get("unsupported_claims", [])
    gold_unsupported = gold.get("unsupported_claims", [])
    # Penalize inventing unsupported claims when gold expects none, and reward
    # matching non-empty unsupported lists when present.
    if not gold_unsupported:
        unsupported_penalty = 1.0 if not predicted_unsupported else 0.0
    else:
        unsupported_penalty = field_accuracy(
            {"unsupported_claims": predicted_unsupported},
            {"unsupported_claims": gold_unsupported},
        ).get("unsupported_claims", 0.0)
    return {
        "answer": answer_score,
        "citation_support": citation_score,
        "unsupported_claims": unsupported_penalty,
        "claim_grounding": (answer_score + citation_score) / 2.0,
    }


def execute_workflow_tool(tool_name: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Shared deterministic tool stubs for executed workflow conditions."""
    request_id = sample.get("request_id") or sample.get("gold", {}).get(
        "extracted_entities", {}
    ).get("request_id", "")
    return {
        "tool": tool_name,
        "status": "ok",
        "request_id": request_id,
        "account_id": sample.get("account_id"),
        "invoice_id": sample.get("invoice_id"),
    }


def workflow_execution_scores(
    predicted: dict[str, Any],
    gold: dict[str, Any],
    *,
    tool_result: dict[str, Any] | None = None,
    state_path: list[str] | None = None,
) -> dict[str, float]:
    """Score classification/tool choice plus executed tool/state evidence."""
    decision = field_accuracy(
        {
            "request_classification": predicted.get("request_classification"),
            "tool_used": predicted.get("tool_used"),
            "requires_human_review": predicted.get("requires_human_review"),
        },
        {
            "request_classification": gold.get("request_classification"),
            "tool_used": gold.get("tool_used"),
            "requires_human_review": gold.get("requires_human_review"),
        },
    )
    expected_tool = gold.get("expected_tool_result", {})
    if tool_result is None:
        tool_ok = 0.0
    else:
        tool_ok = float(
            tool_result.get("tool") == expected_tool.get("tool")
            and tool_result.get("status") == expected_tool.get("status")
            and tool_result.get("request_id") == expected_tool.get("request_id")
        )
    expected_path = list(gold.get("expected_state_path", []))
    if state_path is None:
        state_ok = 0.0
    elif not expected_path:
        state_ok = 1.0
    else:
        state_ok = float(list(state_path) == expected_path)
    decision["tool_execution"] = tool_ok
    decision["state_path"] = state_ok
    return decision


@dataclass
class TrialMetrics:
    schema_valid: bool
    field_scores: dict[str, float]
    retries: int
    latency_ms: float
    tokens_in: int
    tokens_out: int
    error: str | None = None

    @property
    def field_f1(self) -> float:
        if not self.field_scores:
            return 1.0 if self.schema_valid else 0.0
        return sum(self.field_scores.values()) / len(self.field_scores)
