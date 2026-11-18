import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.human_audit import (  # noqa: E402
    pairwise_agreement,
    render_markdown,
    render_summary_markdown,
    review_reasons,
    select_audit_cases,
    summarize_reviews,
    write_packet,
    write_summary,
)


def trial(
    sample_id: str,
    *,
    task: str = "compliance_qa",
    framework: str = "pydantic_ai",
    quality_score: float = 1.0,
    schema_valid: bool = True,
    field_scores: dict[str, float] | None = None,
    retries: int = 0,
    error: str | None = None,
) -> dict:
    return {
        "framework": framework,
        "task": task,
        "sample_id": sample_id,
        "seed": 42,
        "schema_valid": schema_valid,
        "quality_metric": "supported_answer_score",
        "quality_score": quality_score,
        "field_f1": quality_score,
        "field_scores": field_scores or {"answer": quality_score},
        "retries": retries,
        "latency_ms": 100,
        "tokens_in": 20,
        "tokens_out": 10,
        "error": error,
        "raw_output": "Authorization: Bearer abc123",
        "parsed_output": {"answer": "predicted"},
    }


def test_review_reasons_identify_task_specific_cases() -> None:
    assert review_reasons(
        trial("policy", quality_score=0.5, field_scores={"citation_support": 0.0})
    ) == ["low_quality_score", "citation_support"]
    assert review_reasons(
        trial(
            "sql",
            task="sql_generation",
            quality_score=0.4,
            field_scores={"result_correct": 0.0},
        )
    ) == ["low_quality_score", "sql_result_or_execution"]
    assert review_reasons(trial("retry", retries=1)) == ["retry_used"]


def test_select_audit_cases_is_deterministic_by_task() -> None:
    trials = [
        trial("pass", task="compliance_qa", quality_score=1.0, framework="plain_sdk"),
        trial("bad", task="compliance_qa", quality_score=0.1, framework="agno"),
        trial("worse", task="compliance_qa", quality_score=0.0, framework="pydantic_ai"),
        trial("sum-pass", task="summarization", quality_score=1.0),
        trial("sum-bad", task="summarization", quality_score=0.2),
    ]

    packet = select_audit_cases(trials, cases_per_task=1, controls_per_task=1)
    cases = packet["cases"]

    assert packet["n_cases"] == 4
    assert [case["sample_id"] for case in cases] == [
        "worse",
        "pass",
        "sum-bad",
        "sum-pass",
    ]
    assert cases[0]["case_type"] == "diagnostic"
    assert cases[1]["case_type"] == "passing_control"


def test_render_markdown_omits_raw_output_by_default() -> None:
    packet = select_audit_cases([trial("bad", quality_score=0.2)], cases_per_task=1)
    markdown = render_markdown(packet)

    assert "# Human Audit Packet" in markdown
    assert "Parsed output excerpt" in markdown
    assert "Raw output excerpt" not in markdown


def test_write_packet_outputs_json_csv_and_markdown(tmp_path) -> None:
    trials_path = tmp_path / "trials.jsonl"
    json_path = tmp_path / "human_audit.json"
    csv_path = tmp_path / "human_audit.csv"
    markdown_path = tmp_path / "human_audit.md"
    records = [trial("bad", quality_score=0.2)]
    trials_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    packet = write_packet(
        trials_path,
        json_output=json_path,
        csv_output=csv_path,
        markdown_output=markdown_path,
        cases_per_task=1,
        include_raw=True,
    )

    assert packet["n_cases"] == 1
    assert json.loads(json_path.read_text())["cases"][0]["sample_id"] == "bad"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["human_correct"] == ""
    assert "abc123" not in rows[0]["raw_output_excerpt"]
    assert markdown_path.read_text().startswith("# Human Audit Packet")


def test_summarize_reviews_reports_completion_and_agreement() -> None:
    rows = [
        {
            **trial("a", task="compliance_qa"),
            "audit_case_id": "case-a",
            "case_type": "diagnostic",
            "human_correct": "yes",
            "human_supported": "yes",
            "needs_adjudication": "no",
            "human_error_category": "",
            "human_severity": "low",
            "reviewer_id": "r1",
        },
        {
            **trial("a", task="compliance_qa"),
            "audit_case_id": "case-a",
            "case_type": "diagnostic",
            "human_correct": "no",
            "human_supported": "yes",
            "needs_adjudication": "yes",
            "human_error_category": "unsupported",
            "human_severity": "high",
            "reviewer_id": "r2",
        },
        {
            **trial("b", task="sql_generation"),
            "audit_case_id": "case-b",
            "case_type": "passing_control",
            "human_correct": "",
            "human_supported": "",
            "needs_adjudication": "",
            "human_error_category": "",
            "human_severity": "",
            "reviewer_id": "",
        },
    ]

    summary = summarize_reviews(rows)

    assert summary["n_rows"] == 3
    assert summary["n_unique_cases"] == 2
    assert summary["n_completed_rows"] == 2
    assert summary["completion_rate"] == 0.5
    assert summary["human_correct_counts"] == {"no": 1, "yes": 1}
    assert summary["human_correct_yes_rate"] == 0.5
    assert summary["agreement"]["human_correct"]["n_pairs"] == 1
    assert summary["agreement"]["human_correct"]["agreement_rate"] == 0.0
    assert summary["agreement"]["human_supported"]["agreement_rate"] == 1.0
    assert "case-a" in render_summary_markdown(summary)


def test_pairwise_agreement_reports_chance_corrected_kappa() -> None:
    rows = [
        {
            "audit_case_id": "case-1",
            "human_correct": "yes",
            "reviewer_id": "r1",
        },
        {
            "audit_case_id": "case-1",
            "human_correct": "yes",
            "reviewer_id": "r2",
        },
        {
            "audit_case_id": "case-2",
            "human_correct": "yes",
            "reviewer_id": "r1",
        },
        {
            "audit_case_id": "case-2",
            "human_correct": "yes",
            "reviewer_id": "r2",
        },
        {
            "audit_case_id": "case-3",
            "human_correct": "no",
            "reviewer_id": "r1",
        },
        {
            "audit_case_id": "case-3",
            "human_correct": "no",
            "reviewer_id": "r2",
        },
        {
            "audit_case_id": "case-4",
            "human_correct": "no",
            "reviewer_id": "r1",
        },
        {
            "audit_case_id": "case-4",
            "human_correct": "yes",
            "reviewer_id": "r2",
        },
    ]

    agreement = pairwise_agreement(rows, "human_correct")

    assert agreement["n_pairs"] == 4
    assert agreement["agreement_rate"] == 0.75
    assert agreement["chance_agreement"] == 0.5
    assert agreement["cohen_kappa"] == 0.5


def test_write_summary_reads_completed_csv(tmp_path) -> None:
    packet = select_audit_cases([trial("bad", quality_score=0.2)], cases_per_task=1)
    row = packet["cases"][0]
    row["human_correct"] = "yes"
    row["human_supported"] = "yes"
    row["reviewer_id"] = "r1"

    csv_path = tmp_path / "review.csv"
    summary_path = tmp_path / "human_audit_summary.json"
    markdown_path = tmp_path / "human_audit_summary.md"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    summary = write_summary(
        csv_path,
        summary_output=summary_path,
        markdown_output=markdown_path,
    )

    assert summary["n_completed_rows"] == 1
    assert json.loads(summary_path.read_text())["human_supported_yes_rate"] == 1.0
    assert markdown_path.read_text().startswith("# Human Audit Summary")
