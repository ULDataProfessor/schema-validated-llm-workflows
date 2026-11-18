import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.cost_projection import (  # noqa: E402
    build_projection,
    projected_cost,
    render_markdown,
    write_projection,
)


def config():
    return {
        "frameworks": ["pydantic_ai", "dspy"],
        "tasks": ["extraction", "sql_generation"],
        "seeds": [42, 43, 44],
        "sample_limits": {"extraction": 100, "sql_generation": 50},
        "pilot": {"sample_limits": {"extraction": 10, "sql_generation": 10}},
    }


def trial(framework, task, *, tokens_in=100, tokens_out=50):
    return {
        "framework": framework,
        "task": task,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def test_projected_cost_uses_per_million_rates():
    assert (
        projected_cost(
            tokens_in=1000,
            tokens_out=500,
            input_cost_per_million=2.0,
            output_cost_per_million=8.0,
        )
        == 0.006
    )


def test_build_projection_scales_observed_token_means_to_full_matrix():
    projection = build_projection(
        [
            trial("pydantic_ai", "extraction", tokens_in=100, tokens_out=50),
            trial("pydantic_ai", "extraction", tokens_in=200, tokens_out=100),
            trial("pydantic_ai", "sql_generation", tokens_in=50, tokens_out=25),
        ],
        config=config(),
        input_cost_per_million=1.0,
        output_cost_per_million=4.0,
    )

    extraction = projection["cells"]["pydantic_ai::extraction"]
    sql = projection["cells"]["pydantic_ai::sql_generation"]

    assert extraction["target_rows"] == 300
    assert extraction["mean_tokens_in"] == 150.0
    assert extraction["mean_tokens_out"] == 75.0
    assert extraction["projected_tokens_in"] == 45000
    assert extraction["projected_tokens_out"] == 22500
    assert extraction["projected_cost"] == 0.135
    assert sql["target_rows"] == 150
    assert sql["projected_cost"] == 0.0225


def test_build_projection_marks_missing_token_accounting_as_unknown():
    projection = build_projection(
        [trial("dspy", "extraction", tokens_in=0, tokens_out=0)],
        config=config(),
        input_cost_per_million=1.0,
        output_cost_per_million=4.0,
    )

    cell = projection["cells"]["dspy::extraction"]

    assert cell["observed_trials"] == 1
    assert cell["observed_token_trials"] == 0
    assert cell["projection_status"] == "missing_token_accounting"
    assert cell["projected_cost"] is None
    assert "dspy::extraction" in projection["_totals"]["unknown_cells"]


def test_build_projection_rejects_negative_prices():
    try:
        build_projection(
            [trial("pydantic_ai", "extraction")],
            config=config(),
            input_cost_per_million=-1.0,
            output_cost_per_million=4.0,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative prices should fail")


def test_render_markdown_reports_unknown_cells():
    projection = build_projection(
        [trial("dspy", "extraction", tokens_in=0, tokens_out=0)],
        config=config(),
        input_cost_per_million=1.0,
        output_cost_per_million=4.0,
    )

    markdown = render_markdown(projection)

    assert "# Matrix Cost Projection" in markdown
    assert "Unknown cells: `4`" in markdown
    assert "| dspy | extraction | 1 | 0.0% | 300 | -- | -- | -- | missing_token_accounting |" in markdown


def test_write_projection_writes_json_and_markdown(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    config_path = tmp_path / "trials.yaml"
    json_path = tmp_path / "projection.json"
    markdown_path = tmp_path / "projection.md"
    trials_path.write_text(json.dumps(trial("pydantic_ai", "extraction")) + "\n")
    config_path.write_text(
        """
frameworks: [pydantic_ai]
tasks: [extraction]
seeds: [42, 43, 44]
sample_limits:
  extraction: 100
pilot:
  sample_limits:
    extraction: 10
"""
    )

    projection = write_projection(
        trials_path,
        config_path=config_path,
        json_output=json_path,
        markdown_output=markdown_path,
        input_cost_per_million=1.0,
        output_cost_per_million=4.0,
    )

    assert projection["cells"]["pydantic_ai::extraction"]["target_rows"] == 300
    assert json.loads(json_path.read_text())["_pricing"]["currency"] == "USD"
    assert markdown_path.read_text().startswith("# Matrix Cost Projection")
