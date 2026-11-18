"""Tests for hardened SQL, summarization, compliance, and workflow metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    compliance_claim_grounding_scores,
    score_sql_result,
    sql_is_read_only,
    summarization_quality_scores,
    workflow_execution_scores,
    execute_workflow_tool,
)
from eval.schema_boundary import revalidate_output  # noqa: E402
from data.schemas.sql import SQLGenerationResult  # noqa: E402


SETUP = """
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    region TEXT NOT NULL,
    product TEXT NOT NULL,
    amount REAL NOT NULL,
    sale_date TEXT NOT NULL
);
INSERT INTO sales (region, product, amount, sale_date) VALUES ('North', 'Widget Pro', 100.0, '2024-01-01');
INSERT INTO sales (region, product, amount, sale_date) VALUES ('South', 'ReportDash', 200.0, '2024-01-02');
"""


def test_sql_rejects_writes_and_uses_separate_scoring():
    assert not sql_is_read_only("DELETE FROM sales")
    assert sql_is_read_only("SELECT COUNT(*) FROM sales")
    scores = score_sql_result(
        "SELECT COUNT(*) FROM sales",
        "SELECT COUNT(*) FROM sales",
        SETUP,
    )
    assert scores["read_only"] == 1.0
    assert scores["result_correct"] == 1.0
    bad = score_sql_result("DELETE FROM sales", "SELECT COUNT(*) FROM sales", SETUP)
    assert bad["read_only"] == 0.0
    assert bad["result_correct"] == 0.0


def test_summarization_penalizes_unsupported_content():
    gold = {
        "headline": "Q1 revenue $10M with 5% growth",
        "revenue_summary": "Revenue was $10M and growth was 5%.",
        "key_risks": ["supplier delays"],
        "strategic_recommendations": ["expand renewals"],
        "unsupported_forbidden_terms": ["bankruptcy filing"],
    }
    good = {
        "headline": "Q1 revenue $10M with 5% growth",
        "revenue_summary": "Revenue was $10M and growth was 5%.",
        "key_risks": ["supplier delays"],
        "strategic_recommendations": ["expand renewals"],
        "outlook": "ok",
    }
    bad = dict(good)
    bad["outlook"] = "Despite bankruptcy filing, outlook is fine"
    assert summarization_quality_scores(good, gold)["unsupported_content"] == 1.0
    assert summarization_quality_scores(bad, gold)["unsupported_content"] == 0.0


def test_compliance_grounding_and_unsupported_penalty():
    document = "Policy Section: Travel\nBook travel 14 days ahead."
    gold = {
        "answer": "Book travel 14 days ahead.",
        "citations": [{"quote": "Book travel 14 days ahead", "section": "Travel"}],
        "unsupported_claims": [],
    }
    predicted = {
        "answer": "Book travel 14 days ahead.",
        "citations": [{"quote": "Book travel 14 days ahead", "section": "Travel"}],
        "unsupported_claims": ["Employees may ignore the policy"],
    }
    scores = compliance_claim_grounding_scores(predicted, gold, document)
    assert scores["citation_support"] == 1.0
    assert scores["unsupported_claims"] == 0.0


def test_workflow_execution_scores_tool_and_state():
    sample = {"request_id": "REQ-1", "account_id": "A1", "invoice_id": "I1"}
    gold = {
        "request_classification": "billing",
        "tool_used": "lookup_invoice",
        "requires_human_review": False,
        "expected_tool_result": {
            "tool": "lookup_invoice",
            "status": "ok",
            "request_id": "REQ-1",
        },
        "expected_state_path": ["classify", "select_tool", "execute_tool", "finalize"],
    }
    predicted = {
        "request_classification": "billing",
        "tool_used": "lookup_invoice",
        "requires_human_review": False,
    }
    tool_result = execute_workflow_tool("lookup_invoice", sample)
    scores = workflow_execution_scores(
        predicted,
        gold,
        tool_result=tool_result,
        state_path=["classify", "select_tool", "execute_tool", "finalize"],
    )
    assert scores["tool_execution"] == 1.0
    assert scores["state_path"] == 1.0


def test_boundary_revalidation_accepts_valid_sql_payload():
    output = SQLGenerationResult(
        sql="SELECT 1",
        explanation="ok",
        tables_used=["sales"],
    )
    valid, parsed, error = revalidate_output("sql_generation", output)
    assert valid is True
    assert parsed is not None
    assert error is None
