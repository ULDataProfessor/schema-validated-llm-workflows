"""Route trial rows using programmatic checks plus optional judge flags."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.failure_examples import low_field_scores, quality_score
from eval.report import classify_error, load_trials


TRIAL_KEY_FIELDS = ("framework", "task", "sample_id", "seed")


def trial_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in TRIAL_KEY_FIELDS)


def judge_flagged(row: dict[str, Any]) -> bool:
    if row.get("judge_flagged") is not None:
        return bool(row["judge_flagged"])
    judge = row.get("judge")
    if isinstance(judge, dict) and judge.get("flagged") is not None:
        return bool(judge["flagged"])
    return False


def judge_reason(row: dict[str, Any]) -> str:
    if row.get("judge_reason"):
        return str(row["judge_reason"])
    judge = row.get("judge")
    if isinstance(judge, dict):
        return str(judge.get("reason") or judge.get("rationale") or "")
    return ""


def programmatic_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    task = str(row.get("task", ""))
    fields = low_field_scores(row)
    error_type = classify_error(row.get("error"))

    if not bool(row.get("schema_valid")):
        reasons.append("schema")
    if error_type:
        reasons.append(f"error:{error_type}")

    if task == "compliance_qa" and fields.get("citation_support", 1.0) < 1.0:
        reasons.append("evidence")
    if task == "summarization" and any(
        field.startswith("fact_") and score < 1.0 for field, score in fields.items()
    ):
        reasons.append("evidence")
    if task == "sql_generation" and (
        fields.get("execution_success", 1.0) < 1.0
        or fields.get("result_correct", 1.0) < 1.0
        or fields.get("schema_adherence", 1.0) < 1.0
    ):
        reasons.append("evidence")
    if task == "multi_step" and (
        fields.get("tool_used", 1.0) < 1.0
        or fields.get("request_classification", 1.0) < 1.0
        or fields.get("workflow_decision_score", 1.0) < 1.0
    ):
        reasons.append("procedure")
    if task == "multi_step" and (
        fields.get("requires_human_review", 1.0) < 1.0
        or fields.get("final_disposition", 1.0) < 1.0
        or fields.get("state_consistency", 1.0) < 1.0
        or fields.get("procedure_compliance", 1.0) < 1.0
    ):
        reasons.append("state")

    return list(dict.fromkeys(reasons))


def route_trial(row: dict[str, Any]) -> dict[str, Any]:
    reasons = programmatic_reasons(row)
    if judge_flagged(row):
        reasons.append("judge")
    if row.get("requires_human_review") is True:
        reasons.append("workflow")
    reasons = list(dict.fromkeys(reasons))

    if "schema" in reasons:
        route = "reject"
    elif reasons:
        route = "review"
    else:
        route = "accept"

    return {
        "framework": row.get("framework"),
        "task": row.get("task"),
        "sample_id": row.get("sample_id"),
        "seed": row.get("seed"),
        "route": route,
        "reasons": reasons,
        "schema_valid": bool(row.get("schema_valid")),
        "quality_score": quality_score(row),
        "quality_metric": row.get("quality_metric"),
        "judge_flagged": judge_flagged(row),
        "judge_reason": judge_reason(row),
    }


def load_judge_labels(path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    text = path.read_text()
    if path.suffix == ".json":
        loaded = json.loads(text)
        records = loaded["labels"] if isinstance(loaded, dict) and "labels" in loaded else loaded
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    labels: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        labels[trial_key(record)] = record
    return labels


def attach_judge_labels(
    trials: list[dict[str, Any]],
    labels: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for trial in trials:
        row = dict(trial)
        label = labels.get(trial_key(trial))
        if label:
            row["judge_flagged"] = bool(label.get("judge_flagged", label.get("flagged", False)))
            row["judge_reason"] = str(label.get("judge_reason", label.get("reason", "")))
            if label.get("judge_score") is not None:
                row["judge_score"] = label["judge_score"]
        merged.append(row)
    return merged


def build_report(
    trials: list[dict[str, Any]],
    *,
    judge_labels: dict[tuple[str, str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if judge_labels is not None:
        trials = attach_judge_labels(trials, judge_labels)

    routes = [route_trial(trial) for trial in trials]
    route_counts = Counter(route["route"] for route in routes)
    reason_counts = Counter(reason for route in routes for reason in route["reasons"])
    by_task_route = Counter((str(route["task"]), route["route"]) for route in routes)

    return {
        "n_trials": len(trials),
        "n_judge_labels": len(judge_labels or {}),
        "route_counts": dict(sorted(route_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_task_route": {
            f"{task}:{route}": count
            for (task, route), count in sorted(by_task_route.items())
        },
        "routes": routes,
    }


def render_markdown(report: dict[str, Any], *, max_rows: int = 50) -> str:
    lines = [
        "# Judge-Assisted Triage Report",
        "",
        f"- Trial rows inspected: `{report['n_trials']}`",
        f"- Judge labels loaded: `{report['n_judge_labels']}`",
        f"- Route counts: `{report['route_counts']}`",
        f"- Reason counts: `{report['reason_counts']}`",
        "",
        "| Framework | Task | Sample | Route | Reasons | Judge flagged |",
        "|-----------|------|--------|-------|---------|---------------|",
    ]
    for route in report["routes"][:max_rows]:
        reasons = ", ".join(route["reasons"])
        lines.append(
            f"| `{route['framework']}` | `{route['task']}` | `{route['sample_id']}` | "
            f"`{route['route']}` | `{reasons}` | `{route['judge_flagged']}` |"
        )
    if len(report["routes"]) > max_rows:
        lines.append(f"| ... | ... | ... | ... | `{len(report['routes']) - max_rows} more rows` | ... |")
    lines.append("")
    return "\n".join(lines)


def write_report(
    trials_path: Path,
    *,
    judge_labels_path: Path | None = None,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    max_rows: int = 50,
) -> dict[str, Any]:
    labels = load_judge_labels(judge_labels_path) if judge_labels_path else None
    report = build_report(load_trials(trials_path), judge_labels=labels)
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2) + "\n")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(report, max_rows=max_rows))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route trial rows using programmatic checks and optional judge flags"
    )
    parser.add_argument("trials", type=Path, help="Path to trials.jsonl")
    parser.add_argument(
        "--judge-labels",
        type=Path,
        help="Optional JSON or JSONL judge labels keyed by framework/task/sample_id/seed",
    )
    parser.add_argument("--json-output", type=Path, help="Optional JSON report path")
    parser.add_argument("--markdown-output", type=Path, help="Optional Markdown report path")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Maximum routed rows to include in the Markdown table",
    )
    args = parser.parse_args()

    report = write_report(
        args.trials,
        judge_labels_path=args.judge_labels,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        max_rows=args.max_rows,
    )
    print(render_markdown(report, max_rows=args.max_rows))


if __name__ == "__main__":
    main()
