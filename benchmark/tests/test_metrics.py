import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    citation_support_score,
    citation_valid,
    connect_sample_database,
    field_accuracy,
    macro_f1,
    normalize_string,
    phrase_coverage,
    result_sets_equal,
    sql_execution_success,
    sql_syntax_valid,
    summarization_fact_scores,
    token_f1,
)


def test_normalize_string():
    assert normalize_string("  Hello   World  ") == "hello world"


def test_field_accuracy():
    gold = {"vendor_name": "Acme", "total_amount": "100.00"}
    pred = {"vendor_name": "acme", "total_amount": "100.00"}
    scores = field_accuracy(pred, gold)
    assert scores["vendor_name"] == 1.0
    assert scores["total_amount"] == 1.0


def test_field_accuracy_handles_nested_values():
    gold = {"items": [{"sku": "A1", "qty": 2}]}
    pred = {"items": [{"qty": 2, "sku": "A1"}]}
    assert field_accuracy(pred, gold)["items"] == 1.0


def test_macro_f1():
    scores = [{"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 1.0}]
    assert macro_f1(scores) == 0.75


def test_phrase_coverage_accepts_paraphrase_containing_key_phrase():
    predicted = {
        "key_risks": ["Supplier delays may affect fulfillment"],
        "strategic_recommendations": ["Tighten forecasting processes"],
    }

    assert phrase_coverage(predicted, ["supplier delays", "tighten forecasting"]) == 1.0


def test_summarization_fact_scores_check_known_synthetic_facts():
    gold = {
        "headline": "Q3 revenue $21M with 11% growth",
        "revenue_summary": "Revenue was $21M and growth was 11%.",
        "key_risks": ["supplier delays"],
        "strategic_recommendations": ["tighten forecasting"],
    }
    predicted = {
        "headline": "Q3 business review",
        "revenue_summary": "Revenue reached $21 million, up 11%.",
        "key_risks": ["Supplier delays"],
        "strategic_recommendations": ["Tighten forecasting"],
        "outlook": "The quarter depends on supplier risk.",
    }

    assert summarization_fact_scores(predicted, gold) == {
        "fact_quarter": 1.0,
        "fact_revenue": 1.0,
        "fact_growth": 1.0,
        "risk_coverage": 1.0,
        "recommendation_coverage": 1.0,
    }


def test_summarization_fact_scores_support_negative_growth():
    gold = {
        "headline": "Q3 revenue $8M with -2% growth",
        "revenue_summary": "Revenue was $8M and growth was -2%.",
        "key_risks": ["supplier delays"],
        "strategic_recommendations": ["expand renewals"],
    }
    predicted = {
        "headline": "Q3 revenue $8M with -2% growth",
        "revenue_summary": "Revenue was $8M and growth was -2%.",
        "key_risks": ["supplier delays"],
        "strategic_recommendations": ["expand renewals"],
        "outlook": "Manage supplier delays.",
    }

    assert summarization_fact_scores(predicted, gold)["fact_growth"] == 1.0


def test_token_f1_tolerates_short_answer_paraphrase():
    score = token_f1(
        "Employees must report safety incidents to their manager within 24 hours.",
        "Employees must report safety incidents within 24 hours.",
    )

    assert round(score, 2) == 0.93


def test_citation_valid():
    doc = "Employees must report incidents within 24 hours."
    assert citation_valid("report incidents within 24 hours", doc)
    assert not citation_valid("vacation policy", doc)


def test_citation_support_score():
    doc = "Expenses above 500 dollars require written approval."
    citations = [{"quote": "above 500 dollars require written approval"}, {"quote": "missing"}]
    assert citation_support_score(citations, doc) == 0.5


def test_sql_metrics():
    conn = connect_sample_database(
        """
        CREATE TABLE sales (region TEXT, amount REAL);
        INSERT INTO sales VALUES ('North', 10.0);
        INSERT INTO sales VALUES ('South', 5.0);
        """
    )
    try:
        assert sql_syntax_valid("SELECT region, amount FROM sales", conn)
        ok, rows = sql_execution_success("SELECT region, amount FROM sales", conn)
        assert ok
        assert result_sets_equal(rows, [("South", 5.0), ("North", 10.0)])
    finally:
        conn.close()
