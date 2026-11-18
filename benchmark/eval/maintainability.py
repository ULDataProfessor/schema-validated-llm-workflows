"""Generate and summarize maintainability change-exercise worksheets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_FRAMEWORKS = [
    "pydantic_ai",
    "langgraph",
    "dspy",
    "crewai",
    "agno",
    "plain_sdk",
]

CHANGE_EXERCISES = [
    {
        "exercise_id": "complaint_evidence",
        "exercise": (
            "Add an evidence object to complaint extraction, including source quote "
            "and character span."
        ),
    },
    {
        "exercise_id": "compliance_risk_level",
        "exercise": (
            "Add a required risk_level enum to policy answers and route high-risk "
            "answers to review."
        ),
    },
    {
        "exercise_id": "sql_safety_validator",
        "exercise": (
            "Add a SQL safety validator that rejects mutation statements, undeclared "
            "tables, and queries failing EXPLAIN."
        ),
    },
    {
        "exercise_id": "workflow_review_disposition",
        "exercise": "Add an explicit review disposition to the multi-step workflow output.",
    },
]

SCORE_FIELDS = [
    "contract_clarity",
    "error_handling",
    "traceability",
    "change_effort",
    "testability",
]

WORKSHEET_FIELDS = [
    "framework",
    "exercise_id",
    "exercise",
    "completed",
    "files_changed",
    "lines_changed",
    "elapsed_minutes",
    "tests_failed_count",
    "validation_failures_before_prompt_repair",
    "prompt_changed",
    "schema_changed",
    "runner_changed",
    "evaluator_changed",
    "reporting_changed",
    "audit_record_changed",
    *SCORE_FIELDS,
    "reviewer_id",
    "notes",
]


def blank_rows(frameworks: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for framework in frameworks or DEFAULT_FRAMEWORKS:
        for exercise in CHANGE_EXERCISES:
            row = {field: "" for field in WORKSHEET_FIELDS}
            row.update(
                {
                    "framework": framework,
                    "exercise_id": exercise["exercise_id"],
                    "exercise": exercise["exercise"],
                }
            )
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WORKSHEET_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _score(value: Any) -> float | None:
    score = _number(value)
    if score is None:
        return None
    if not 1 <= score <= 5:
        raise ValueError(f"Maintainability score must be between 1 and 5: {value}")
    return score


def _is_completed(row: dict[str, Any]) -> bool:
    value = str(row.get("completed", "")).strip().lower()
    return value in {"1", "true", "yes", "y", "complete", "completed"}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_framework: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        framework = str(row.get("framework", "")).strip()
        if framework:
            by_framework[framework].append(row)

    frameworks: dict[str, dict[str, Any]] = {}
    for framework in sorted(by_framework):
        framework_rows = by_framework[framework]
        completed_rows = [row for row in framework_rows if _is_completed(row)]
        score_values = {
            field: [
                score
                for row in completed_rows
                if (score := _score(row.get(field))) is not None
            ]
            for field in SCORE_FIELDS
        }
        dimension_means = {
            field: round(mean(values), 3) if values else None
            for field, values in score_values.items()
        }
        all_scores = [
            score
            for values in score_values.values()
            for score in values
        ]
        files_changed = [
            value
            for row in completed_rows
            if (value := _number(row.get("files_changed"))) is not None
        ]
        lines_changed = [
            value
            for row in completed_rows
            if (value := _number(row.get("lines_changed"))) is not None
        ]
        elapsed_minutes = [
            value
            for row in completed_rows
            if (value := _number(row.get("elapsed_minutes"))) is not None
        ]

        frameworks[framework] = {
            "n_rows": len(framework_rows),
            "n_completed": len(completed_rows),
            "completion_rate": round(len(completed_rows) / len(framework_rows), 3)
            if framework_rows
            else 0.0,
            "mean_files_changed": round(mean(files_changed), 3) if files_changed else None,
            "mean_lines_changed": round(mean(lines_changed), 3) if lines_changed else None,
            "mean_elapsed_minutes": round(mean(elapsed_minutes), 3)
            if elapsed_minutes
            else None,
            "dimension_means": dimension_means,
            "aggregate_score": round(mean(all_scores), 3) if all_scores else None,
        }

    return {
        "n_rows": len(rows),
        "n_frameworks": len(frameworks),
        "score_fields": SCORE_FIELDS,
        "frameworks": frameworks,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Maintainability Change-Exercise Summary",
        "",
        f"- Rows: `{summary['n_rows']}`",
        f"- Frameworks: `{summary['n_frameworks']}`",
        "",
        "| Framework | Completed | Aggregate | Files | Lines | Minutes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for framework, stats in summary["frameworks"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    framework,
                    f"{stats['n_completed']}/{stats['n_rows']}",
                    "--"
                    if stats["aggregate_score"] is None
                    else f"{stats['aggregate_score']:.3f}",
                    "--"
                    if stats["mean_files_changed"] is None
                    else f"{stats['mean_files_changed']:.3f}",
                    "--"
                    if stats["mean_lines_changed"] is None
                    else f"{stats['mean_lines_changed']:.3f}",
                    "--"
                    if stats["mean_elapsed_minutes"] is None
                    else f"{stats['mean_elapsed_minutes']:.3f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Dimension Means", ""])
    for framework, stats in summary["frameworks"].items():
        lines.append(f"### {framework}")
        for field, value in stats["dimension_means"].items():
            rendered = "--" if value is None else f"{value:.3f}"
            lines.append(f"- `{field}`: {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_worksheet(
    *,
    csv_output: Path,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    frameworks: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = blank_rows(frameworks)
    write_csv(csv_output, rows)
    if json_output is not None:
        write_json(json_output, {"rows": rows})
    if markdown_output is not None:
        summary = summarize_rows(rows)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(summary))
    return rows


def write_summary(
    csv_input: Path,
    *,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
) -> dict[str, Any]:
    summary = summarize_rows(read_csv(csv_input))
    if json_output is not None:
        write_json(json_output, summary)
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or summarize maintainability change-exercise worksheets"
    )
    parser.add_argument("csv_input", type=Path, nargs="?", help="Completed worksheet CSV")
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--framework", action="append", dest="frameworks")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        if args.csv_input is None:
            raise SystemExit("--summarize requires a completed worksheet CSV")
        summary = write_summary(
            args.csv_input,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
        )
        if args.json_output is None and args.markdown_output is None:
            print(json.dumps(summary, indent=2))
        return

    if args.csv_output is None:
        raise SystemExit("Worksheet generation requires --csv-output")
    rows = write_worksheet(
        csv_output=args.csv_output,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        frameworks=args.frameworks,
    )
    print(f"Wrote {len(rows)} worksheet rows to {args.csv_output}")


if __name__ == "__main__":
    main()
