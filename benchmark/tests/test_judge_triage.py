import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.judge_triage import (  # noqa: E402
    build_report,
    load_judge_labels,
    programmatic_reasons,
    render_markdown,
    route_trial,
    write_report,
)


def trial(
    sample_id: str,
    *,
    task: str = "compliance_qa",
    schema_valid: bool = True,
    quality_score: float = 1.0,
    field_scores: dict[str, float] | None = None,
    error: str | None = None,
    requires_human_review: bool | None = None,
) -> dict:
    row = {
        "framework": "pydantic_ai",
        "task": task,
        "sample_id": sample_id,
        "seed": 42,
        "schema_valid": schema_valid,
        "quality_metric": "supported_answer_score",
        "quality_score": quality_score,
        "field_scores": field_scores or {"answer": quality_score},
        "retries": 0,
        "latency_ms": 100,
        "tokens_in": 10,
        "tokens_out": 5,
        "error": error,
        "parsed_output": {"answer": "ok"},
    }
    if requires_human_review is not None:
        row["requires_human_review"] = requires_human_review
    return row


def test_programmatic_reasons_detect_task_specific_hard_stops() -> None:
    assert programmatic_reasons(trial("schema", schema_valid=False)) == ["schema"]
    assert programmatic_reasons(
        trial("policy", field_scores={"citation_support": 0.0})
    ) == ["evidence"]
    assert programmatic_reasons(
        trial("sql", task="sql_generation", field_scores={"result_correct": 0.0})
    ) == ["evidence"]
    assert programmatic_reasons(
        trial(
            "workflow",
            task="multi_step",
            field_scores={"tool_used": 0.0, "requires_human_review": 0.0},
        )
    ) == ["procedure", "state"]


def test_route_trial_rejects_schema_and_reviews_judge_flags() -> None:
    rejected = route_trial(trial("schema", schema_valid=False))
    reviewed = route_trial({**trial("judge"), "judge_flagged": True})
    accepted = route_trial(trial("pass"))

    assert rejected["route"] == "reject"
    assert rejected["reasons"] == ["schema"]
    assert reviewed["route"] == "review"
    assert reviewed["reasons"] == ["judge"]
    assert accepted["route"] == "accept"


def test_build_report_joins_judge_labels_and_counts_routes() -> None:
    rows = [trial("pass"), trial("needs-judge")]
    labels = {
        ("pydantic_ai", "compliance_qa", "needs-judge", "42"): {
            "flagged": True,
            "reason": "possible unsupported answer",
        }
    }

    report = build_report(rows, judge_labels=labels)

    assert report["n_judge_labels"] == 1
    assert report["route_counts"] == {"accept": 1, "review": 1}
    assert report["reason_counts"] == {"judge": 1}
    reviewed = [route for route in report["routes"] if route["route"] == "review"][0]
    assert reviewed["judge_reason"] == "possible unsupported answer"


def test_load_judge_labels_reads_jsonl(tmp_path: Path) -> None:
    labels_path = tmp_path / "judge_labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "framework": "pydantic_ai",
                "task": "compliance_qa",
                "sample_id": "a",
                "seed": 42,
                "flagged": True,
            }
        )
        + "\n"
    )

    labels = load_judge_labels(labels_path)

    assert labels[("pydantic_ai", "compliance_qa", "a", "42")]["flagged"] is True


def test_write_report_outputs_json_and_markdown(tmp_path: Path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    json_output = tmp_path / "judge_triage.json"
    markdown_output = tmp_path / "judge_triage.md"
    trials_path.write_text(json.dumps(trial("review", field_scores={"citation_support": 0.0})) + "\n")

    report = write_report(
        trials_path,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    assert report["route_counts"] == {"review": 1}
    assert json.loads(json_output.read_text())["reason_counts"] == {"evidence": 1}
    markdown = markdown_output.read_text()
    assert markdown.startswith("# Judge-Assisted Triage Report")
    assert "`review`" in markdown
    assert "Judge-Assisted Triage Report" in render_markdown(report)
