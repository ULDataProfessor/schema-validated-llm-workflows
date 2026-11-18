import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.failure_examples import (  # noqa: E402
    failure_categories,
    redact,
    render_markdown,
    select_examples,
    write_examples,
)


def trial(
    sample_id: str,
    *,
    framework: str = "pydantic_ai",
    task: str = "extraction",
    schema_valid: bool = True,
    quality_score: float = 1.0,
    retries: int = 0,
    error: str | None = None,
    field_scores: dict[str, float] | None = None,
) -> dict:
    return {
        "framework": framework,
        "task": task,
        "sample_id": sample_id,
        "seed": 42,
        "schema_valid": schema_valid,
        "quality_metric": "field_macro_accuracy",
        "quality_score": quality_score,
        "field_f1": quality_score,
        "field_scores": field_scores or {"answer": quality_score},
        "retries": retries,
        "latency_ms": 100,
        "tokens_in": 20,
        "tokens_out": 10,
        "error": error,
        "raw_output": "raw text",
        "parsed_output": {"answer": "predicted"},
    }


def test_failure_categories_cover_taxonomy_cases():
    assert failure_categories(trial("parse", error="JSON parse failed")) == [
        "parse_or_transport_failure"
    ]
    assert failure_categories(trial("schema", schema_valid=False)) == [
        "schema_violation"
    ]
    assert failure_categories(trial("semantic", quality_score=0.2)) == [
        "schema_valid_semantic_error"
    ]
    assert failure_categories(
        trial(
            "tool",
            task="multi_step",
            quality_score=0.5,
            field_scores={"tool_used": 0.0},
        )
    ) == ["schema_valid_semantic_error", "tool_or_orchestration_error"]
    assert failure_categories(
        trial(
            "state",
            task="multi_step",
            quality_score=0.5,
            field_scores={"requires_human_review": 0.0},
        )
    ) == ["schema_valid_semantic_error", "state_or_procedure_failure"]
    assert failure_categories(
        trial("retry", schema_valid=False, retries=3, error="retry budget exceeded")
    ) == ["retry_exhaustion", "schema_violation"]
    assert failure_categories(trial("recovered", retries=1)) == ["recovered_retry"]


def test_select_examples_is_deterministic_and_limits_per_category():
    trials = [
        trial("a", quality_score=0.6, framework="plain_sdk"),
        trial("b", quality_score=0.2, framework="pydantic_ai"),
        trial("c", quality_score=0.4, framework="agno"),
    ]

    report = select_examples(trials, per_category=2, semantic_threshold=0.99)
    examples = report["examples"]["schema_valid_semantic_error"]

    assert report["category_counts"]["schema_valid_semantic_error"] == 3
    assert [example["sample_id"] for example in examples] == ["b", "c"]


def test_redact_removes_common_secret_patterns():
    text = "Authorization: Bearer abc123 token=secret password=hunter2"

    redacted = redact(text)

    assert "abc123" not in redacted
    assert "secret" not in redacted
    assert "hunter2" not in redacted


def test_render_markdown_omits_raw_output_by_default():
    report = select_examples([trial("a", quality_score=0.2)], per_category=1)
    markdown = render_markdown(report)

    assert "# Qualitative Failure Examples" in markdown
    assert "Schema-valid semantic error" in markdown
    assert "Parsed output excerpt" in markdown
    assert "Raw output excerpt" not in markdown


def test_write_examples_writes_markdown_and_json(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    markdown_path = tmp_path / "examples.md"
    json_path = tmp_path / "examples.json"
    records = [trial("a", quality_score=0.2)]
    trials_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    report = write_examples(
        trials_path,
        markdown_output=markdown_path,
        json_output=json_path,
        per_category=1,
    )

    assert report["n_trials"] == 1
    assert markdown_path.exists()
    assert json.loads(json_path.read_text())["category_counts"] == {
        "schema_valid_semantic_error": 1
    }
