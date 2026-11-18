import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.cost_analysis import (  # noqa: E402
    aggregate_costs,
    quality_per_1k_tokens,
    render_markdown,
    trial_cost,
    write_cost_analysis,
)


def trial(
    framework: str,
    *,
    task: str = "extraction",
    schema_valid: bool = True,
    retries: int = 0,
    quality_score: float = 1.0,
    tokens_in: int = 1000,
    tokens_out: int = 500,
) -> dict:
    return {
        "framework": framework,
        "task": task,
        "schema_valid": schema_valid,
        "retries": retries,
        "quality_score": quality_score,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def test_trial_cost_uses_input_and_output_rates_per_million():
    record = trial("pydantic_ai", tokens_in=1000, tokens_out=500)

    cost = trial_cost(record, input_cost_per_million=2.0, output_cost_per_million=8.0)

    assert cost == 0.006


def test_aggregate_costs_reports_group_efficiency_metrics():
    records = [
        trial("pydantic_ai", quality_score=1.0, tokens_in=1000, tokens_out=500),
        trial(
            "pydantic_ai",
            schema_valid=False,
            retries=2,
            quality_score=0.0,
            tokens_in=500,
            tokens_out=500,
        ),
    ]

    summary = aggregate_costs(
        records,
        input_cost_per_million=2.0,
        output_cost_per_million=8.0,
    )
    stats = summary["pydantic_ai::extraction"]

    assert stats["n"] == 2
    assert stats["total_tokens_in"] == 1500
    assert stats["total_tokens_out"] == 1000
    assert stats["token_accounting_coverage_rate"] == 1.0
    assert stats["total_estimated_cost"] == 0.011
    assert stats["schema_valid_successes"] == 1
    assert stats["zero_retry_successes"] == 1
    assert stats["cost_per_schema_valid_run"] == 0.011
    assert stats["mean_quality_score"] == 0.5
    assert stats["quality_per_1k_tokens"] == 0.4
    assert stats["quality_points_per_cost"] == round(1.0 / 0.011, 4)


def test_quality_per_1k_tokens_handles_missing_or_zero_tokens():
    assert quality_per_1k_tokens(None, 100) is None
    assert quality_per_1k_tokens(1.0, 0) is None
    assert quality_per_1k_tokens(0.5, 250) == 2.0


def test_aggregate_costs_rejects_negative_prices():
    try:
        aggregate_costs(
            [trial("pydantic_ai")],
            input_cost_per_million=-1.0,
            output_cost_per_million=1.0,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative prices should fail")


def test_aggregate_costs_marks_all_zero_token_groups_as_unknown_cost():
    summary = aggregate_costs(
        [trial("dspy", tokens_in=0, tokens_out=0)],
        input_cost_per_million=2.0,
        output_cost_per_million=8.0,
    )
    stats = summary["dspy::extraction"]

    assert stats["token_accounting_coverage_rate"] == 0.0
    assert stats["total_estimated_cost"] is None
    assert stats["mean_estimated_cost"] is None
    assert stats["cost_per_schema_valid_run"] is None
    assert stats["quality_points_per_cost"] is None


def test_render_markdown_includes_pricing_and_rows():
    summary = aggregate_costs(
        [trial("pydantic_ai")],
        input_cost_per_million=2.0,
        output_cost_per_million=8.0,
        currency="USD",
    )

    markdown = render_markdown(summary)

    assert "# Cost Analysis" in markdown
    assert "Input cost per million tokens: `2.0`" in markdown
    assert "| pydantic_ai | extraction | 1 | 100.0% | 1500 | 0.00600000" in markdown


def test_write_cost_analysis_writes_json_and_markdown(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    json_path = tmp_path / "costs.json"
    markdown_path = tmp_path / "costs.md"
    trials_path.write_text(json.dumps(trial("plain_sdk")) + "\n")

    summary = write_cost_analysis(
        trials_path,
        json_output=json_path,
        markdown_output=markdown_path,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    assert summary["plain_sdk::extraction"]["total_estimated_cost"] == 0.002
    assert json.loads(json_path.read_text())["_pricing"]["currency"] == "USD"
    assert markdown_path.read_text().startswith("# Cost Analysis")
