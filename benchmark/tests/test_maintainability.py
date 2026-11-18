import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.maintainability import (  # noqa: E402
    CHANGE_EXERCISES,
    blank_rows,
    render_markdown,
    summarize_rows,
    write_summary,
    write_worksheet,
)


def test_blank_rows_create_framework_exercise_grid() -> None:
    rows = blank_rows(["pydantic_ai", "plain_sdk"])

    assert len(rows) == 2 * len(CHANGE_EXERCISES)
    assert rows[0]["framework"] == "pydantic_ai"
    assert rows[0]["exercise_id"] == "complaint_evidence"
    assert rows[0]["completed"] == ""


def test_summarize_rows_reports_effort_and_dimension_scores() -> None:
    rows = [
        {
            "framework": "pydantic_ai",
            "exercise_id": "complaint_evidence",
            "completed": "yes",
            "files_changed": "3",
            "lines_changed": "28",
            "elapsed_minutes": "20",
            "contract_clarity": "5",
            "error_handling": "4",
            "traceability": "5",
            "change_effort": "4",
            "testability": "5",
        },
        {
            "framework": "pydantic_ai",
            "exercise_id": "sql_safety_validator",
            "completed": "no",
            "files_changed": "99",
            "lines_changed": "99",
            "elapsed_minutes": "99",
            "contract_clarity": "1",
        },
        {
            "framework": "plain_sdk",
            "exercise_id": "complaint_evidence",
            "completed": "yes",
            "files_changed": "5",
            "lines_changed": "46",
            "elapsed_minutes": "35",
            "contract_clarity": "3",
            "error_handling": "3",
            "traceability": "3",
            "change_effort": "2",
            "testability": "4",
        },
    ]

    summary = summarize_rows(rows)

    assert summary["n_rows"] == 3
    assert summary["frameworks"]["pydantic_ai"]["n_completed"] == 1
    assert summary["frameworks"]["pydantic_ai"]["mean_files_changed"] == 3.0
    assert summary["frameworks"]["pydantic_ai"]["aggregate_score"] == 4.6
    assert summary["frameworks"]["plain_sdk"]["dimension_means"]["change_effort"] == 2.0


def test_summarize_rows_rejects_scores_outside_rubric_range() -> None:
    rows = [
        {
            "framework": "pydantic_ai",
            "exercise_id": "complaint_evidence",
            "completed": "yes",
            "contract_clarity": "6",
        }
    ]

    try:
        summarize_rows(rows)
    except ValueError as exc:
        assert "between 1 and 5" in str(exc)
    else:
        raise AssertionError("scores outside 1..5 should fail")


def test_write_worksheet_and_summary_artifacts(tmp_path) -> None:
    worksheet_csv = tmp_path / "maintainability.csv"
    worksheet_json = tmp_path / "maintainability.json"
    worksheet_md = tmp_path / "maintainability.md"

    rows = write_worksheet(
        csv_output=worksheet_csv,
        json_output=worksheet_json,
        markdown_output=worksheet_md,
        frameworks=["pydantic_ai"],
    )

    assert worksheet_csv.exists()
    assert json.loads(worksheet_json.read_text())["rows"][0]["framework"] == "pydantic_ai"
    assert worksheet_md.read_text().startswith("# Maintainability Change-Exercise Summary")
    assert len(rows) == len(CHANGE_EXERCISES)

    completed = rows[0]
    completed.update(
        {
            "completed": "yes",
            "files_changed": "2",
            "lines_changed": "12",
            "elapsed_minutes": "15",
            "contract_clarity": "5",
            "error_handling": "5",
            "traceability": "4",
            "change_effort": "4",
            "testability": "5",
        }
    )
    with worksheet_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = tmp_path / "maintainability_summary.json"
    summary_md = tmp_path / "maintainability_summary.md"
    summary = write_summary(
        worksheet_csv,
        json_output=summary_json,
        markdown_output=summary_md,
    )

    assert summary["frameworks"]["pydantic_ai"]["n_completed"] == 1
    assert json.loads(summary_json.read_text())["frameworks"]["pydantic_ai"][
        "aggregate_score"
    ] == 4.6
    assert "pydantic_ai" in render_markdown(summary)
