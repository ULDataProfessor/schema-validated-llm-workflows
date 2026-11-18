import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rescore_trials import rescore_record  # noqa: E402


def test_rescore_record_updates_summarization_quality_for_negative_growth():
    record = {
        "framework": "dry_run",
        "task": "summarization",
        "sample_id": "sum_0002",
        "schema_valid": True,
        "parsed_output": {
            "headline": "Q3 revenue $8M with -2% growth",
            "revenue_summary": "Revenue was $8M and growth was -2%.",
            "key_risks": ["supplier delays"],
            "strategic_recommendations": ["expand renewals"],
            "outlook": "Manage supplier delays.",
        },
    }
    sample_index = {
        "summarization": {
            "sum_0002": {
                "id": "sum_0002",
                "gold": {
                    "headline": "Q3 revenue $8M with -2% growth",
                    "revenue_summary": "Revenue was $8M and growth was -2%.",
                    "key_risks": ["supplier delays"],
                    "strategic_recommendations": ["expand renewals"],
                    "outlook": "Outlook depends on whether the team can manage supplier delays.",
                },
            }
        }
    }

    rescored = rescore_record(record, sample_index)

    assert rescored["quality_metric"] == "source_fact_coverage"
    assert rescored["quality_score"] == 1.0
    assert rescored["field_scores"]["fact_growth"] == 1.0
