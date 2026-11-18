import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.schemas.compliance import PolicyAnswer  # noqa: E402
from data.schemas.sql import SQLGenerationResult  # noqa: E402
from data.schemas.summarization import ExecutiveSummary  # noqa: E402
from data.schemas.workflow import WorkflowResult  # noqa: E402
from frameworks.base import DryRunRunner, RunResult  # noqa: E402
from scripts.run_benchmark import configured_task_limits, make_run_id  # noqa: E402


def test_dry_run_sql_generation_returns_schema_instance():
    runner = DryRunRunner()
    sample = {
        "gold": {
            "sql": "SELECT COUNT(*) FROM sales",
            "explanation": "Count rows.",
            "tables_used": ["sales"],
        }
    }

    result = asyncio.run(runner.run_sql_generation(sample))

    assert result.schema_valid
    assert isinstance(result.output, SQLGenerationResult)


def test_dry_run_summarization_returns_schema_instance():
    runner = DryRunRunner()
    sample = {
        "gold": {
            "headline": "Q1 revenue improved",
            "revenue_summary": "Revenue rose 10%.",
            "key_risks": ["churn"],
            "strategic_recommendations": ["expand renewals"],
            "outlook": "Positive with churn risk.",
        }
    }

    result = asyncio.run(runner.run_summarization(sample))

    assert result.schema_valid
    assert isinstance(result.output, ExecutiveSummary)


def test_dry_run_compliance_returns_schema_instance():
    runner = DryRunRunner()
    sample = {
        "gold": {
            "answer": "Report incidents within 24 hours.",
            "citations": [{"quote": "within 24 hours", "section": "Incident Reporting"}],
            "confidence": 0.95,
            "unsupported_claims": [],
        }
    }

    result = asyncio.run(runner.run_compliance_qa(sample))

    assert result.schema_valid
    assert isinstance(result.output, PolicyAnswer)


def test_dry_run_workflow_returns_schema_instance():
    runner = DryRunRunner()
    sample = {
        "gold": {
            "request_classification": "billing",
            "tool_used": "lookup_invoice",
            "extracted_entities": {"request_id": "REQ-1000"},
            "recommendation": "Route to lookup_invoice.",
            "requires_human_review": False,
        }
    }

    result = asyncio.run(runner.run_multi_step(sample))

    assert result.schema_valid
    assert isinstance(result.output, WorkflowResult)


def test_trial_record_includes_parsed_output():
    runner = DryRunRunner()
    sample = {
        "gold": {
            "sql": "SELECT COUNT(*) FROM sales",
            "explanation": "Count rows.",
            "tables_used": ["sales"],
        }
    }
    result = asyncio.run(runner.run_sql_generation(sample))
    result.field_scores = {
        "syntax_valid": 1.0,
        "execution_success": 1.0,
        "result_correct": 1.0,
    }

    record = runner.to_trial_record(
        task="sql_generation",
        sample_id="sql_001",
        seed=42,
        result=result,
    )

    assert record["parsed_output"] == sample["gold"]
    assert record["raw_output"]
    assert record["quality_score"] == 1.0
    assert record["quality_metric"] == "result_correct"


def test_trial_record_preserves_trace_events():
    runner = DryRunRunner()
    result = RunResult(
        output=None,
        schema_valid=False,
        error="tool authorization failed",
        trace_events=[
            {
                "step": 1,
                "kind": "tool_call",
                "tool_name": "run_sql",
                "decision": "refuse",
                "validated": False,
            }
        ],
    )

    record = runner.to_trial_record(
        task="multi_step",
        sample_id="wf_0001",
        seed=42,
        result=result,
    )

    assert record["trace_events"] == result.trace_events


def test_configured_task_limits_supports_pilot_and_override():
    trial_cfg = {
        "sample_limits": {"extraction": 100, "sql_generation": 50},
        "pilot": {"sample_limits": {"extraction": 10, "sql_generation": 5}},
    }
    tasks = ["extraction", "sql_generation"]

    assert configured_task_limits(
        trial_cfg,
        tasks,
        all_tasks=True,
        pilot=False,
        override_limit=None,
    ) == {"extraction": 100, "sql_generation": 50}
    assert configured_task_limits(
        trial_cfg,
        tasks,
        all_tasks=True,
        pilot=True,
        override_limit=None,
    ) == {"extraction": 10, "sql_generation": 5}
    assert configured_task_limits(
        trial_cfg,
        tasks,
        all_tasks=True,
        pilot=True,
        override_limit=2,
    ) == {"extraction": 2, "sql_generation": 2}


def test_make_run_id_is_unique_with_fixed_timestamp():
    now = datetime(2026, 6, 12, 21, 50, 51, tzinfo=timezone.utc)

    run_ids = {make_run_id(now) for _ in range(20)}

    assert len(run_ids) == 20
    assert all(run_id.startswith("20260612T215051000000Z-") for run_id in run_ids)
