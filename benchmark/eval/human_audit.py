"""Create deterministic human-review packets from benchmark trial records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from eval.failure_examples import excerpt, low_field_scores, quality_score
from eval.report import classify_error, load_trials


TASK_REVIEW_FOCUS = {
    "extraction": "field correctness against source document and gold labels",
    "sql_generation": "query intent, safety, execution result, and schema adherence",
    "summarization": "source-fact support, omitted risks, and unsupported claims",
    "compliance_qa": "answer support, citation sufficiency, and confidence calibration",
    "multi_step": "classification, tool choice, review routing, and final disposition",
}

REVIEW_FIELDS = {
    "human_correct": "",
    "human_supported": "",
    "human_error_category": "",
    "human_severity": "",
    "needs_adjudication": "",
    "reviewer_id": "",
    "reviewer_notes": "",
}


def trial_key(trial: dict[str, Any]) -> str:
    return "::".join(
        [
            str(trial.get("framework", "")),
            str(trial.get("task", "")),
            str(trial.get("sample_id", "")),
            str(trial.get("seed", "")),
        ]
    )


def review_reasons(
    trial: dict[str, Any],
    *,
    semantic_threshold: float = 0.99,
) -> list[str]:
    reasons: list[str] = []
    schema_valid = bool(trial.get("schema_valid"))
    error_type = classify_error(trial.get("error"))
    quality = quality_score(trial)
    low_scores = low_field_scores(trial)
    task = str(trial.get("task", ""))
    retries = int(trial.get("retries", 0) or 0)

    if not schema_valid:
        reasons.append("schema_invalid")
    if error_type:
        reasons.append(f"error:{error_type}")
    if quality is not None and quality < semantic_threshold:
        reasons.append("low_quality_score")
    if retries > 0:
        reasons.append("retry_used")

    if task == "compliance_qa" and low_scores.get("citation_support", 1.0) < 1.0:
        reasons.append("citation_support")
    if task == "sql_generation" and (
        low_scores.get("result_correct", 1.0) < 1.0
        or low_scores.get("execution_success", 1.0) < 1.0
    ):
        reasons.append("sql_result_or_execution")
    if task == "summarization" and any(
        field.startswith("fact_") and score < 1.0 for field, score in low_scores.items()
    ):
        reasons.append("source_fact_support")
    if task == "multi_step" and (
        low_scores.get("tool_used", 1.0) < 1.0
        or low_scores.get("requires_human_review", 1.0) < 1.0
    ):
        reasons.append("workflow_decision")

    return list(dict.fromkeys(reasons))


def audit_priority(
    trial: dict[str, Any],
    *,
    semantic_threshold: float = 0.99,
) -> tuple:
    reasons = review_reasons(trial, semantic_threshold=semantic_threshold)
    quality = quality_score(trial)
    quality_value = quality if quality is not None else 1.0
    retries = int(trial.get("retries", 0) or 0)
    error_type = classify_error(trial.get("error"))
    severity = 0

    if not bool(trial.get("schema_valid")):
        severity += 100
    if error_type:
        severity += 80
    if quality is not None and quality < semantic_threshold:
        severity += 60 + int((semantic_threshold - quality_value) * 100)
    if any(reason in reasons for reason in {"citation_support", "workflow_decision"}):
        severity += 40
    if any(reason in reasons for reason in {"sql_result_or_execution", "source_fact_support"}):
        severity += 30
    severity += min(retries * 5, 20)

    return (
        -severity,
        quality_value,
        -retries,
        str(trial.get("framework", "")),
        str(trial.get("sample_id", "")),
        str(trial.get("seed", "")),
    )


def control_priority(trial: dict[str, Any]) -> tuple:
    return (
        str(trial.get("framework", "")),
        str(trial.get("sample_id", "")),
        str(trial.get("seed", "")),
    )


def audit_case(
    trial: dict[str, Any],
    *,
    case_type: str,
    semantic_threshold: float,
    include_raw: bool,
    max_chars: int,
    blind_framework: bool = True,
) -> dict[str, Any]:
    case = {
        "audit_case_id": trial_key(trial),
        "case_type": case_type,
        "review_focus": TASK_REVIEW_FOCUS.get(str(trial.get("task", "")), "task correctness"),
        "review_reasons": review_reasons(trial, semantic_threshold=semantic_threshold),
        "task": trial.get("task"),
        "sample_id": trial.get("sample_id"),
        "seed": trial.get("seed"),
        "schema_valid": bool(trial.get("schema_valid")),
        "quality_metric": trial.get("quality_metric"),
        "quality_score": quality_score(trial),
        "retries": int(trial.get("retries", 0) or 0),
        "latency_ms": trial.get("latency_ms"),
        "tokens_total": (trial.get("tokens_in", 0) or 0)
        + (trial.get("tokens_out", 0) or 0),
        "error_type": classify_error(trial.get("error")),
        "low_field_scores": low_field_scores(trial),
        "field_scores": trial.get("field_scores") or {},
        "parsed_output_excerpt": excerpt(trial.get("parsed_output"), max_chars=max_chars),
        "raw_output_excerpt": "",
        "source_lookup": (
            "Join task and sample_id against benchmark/data/synthetic fixtures or "
            "the documented public-data source before review."
        ),
    }
    if blind_framework:
        case["framework"] = "BLINDED"
        case["framework_key_hash"] = abs(hash(str(trial.get("framework", "")))) % 10_000_000
    else:
        case["framework"] = trial.get("framework")
    if include_raw:
        case["raw_output_excerpt"] = excerpt(trial.get("raw_output"), max_chars=max_chars)
    case.update(REVIEW_FIELDS)
    return case


def select_audit_cases(
    trials: list[dict[str, Any]],
    *,
    cases_per_task: int = 5,
    controls_per_task: int = 1,
    semantic_threshold: float = 0.99,
    include_raw: bool = False,
    max_chars: int = 1200,
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_task[str(trial.get("task", ""))].append(trial)

    cases: list[dict[str, Any]] = []
    selected: set[str] = set()
    for task in sorted(by_task):
        task_trials = by_task[task]
        diagnostics = sorted(
            [
                trial
                for trial in task_trials
                if review_reasons(trial, semantic_threshold=semantic_threshold)
            ],
            key=lambda trial: audit_priority(trial, semantic_threshold=semantic_threshold),
        )
        for trial in diagnostics[:cases_per_task]:
            selected.add(trial_key(trial))
            cases.append(
                audit_case(
                    trial,
                    case_type="diagnostic",
                    semantic_threshold=semantic_threshold,
                    include_raw=include_raw,
                    max_chars=max_chars,
                )
            )

        controls = sorted(
            [
                trial
                for trial in task_trials
                if trial_key(trial) not in selected
                and not review_reasons(trial, semantic_threshold=semantic_threshold)
            ],
            key=control_priority,
        )
        for trial in controls[:controls_per_task]:
            selected.add(trial_key(trial))
            cases.append(
                audit_case(
                    trial,
                    case_type="passing_control",
                    semantic_threshold=semantic_threshold,
                    include_raw=include_raw,
                    max_chars=max_chars,
                )
            )

    return {
        "n_trials": len(trials),
        "n_cases": len(cases),
        "cases_per_task": cases_per_task,
        "controls_per_task": controls_per_task,
        "semantic_threshold": semantic_threshold,
        "include_raw": include_raw,
        "blind_framework": True,
        "review_fields": list(REVIEW_FIELDS),
        "cases": cases,
    }


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Human Audit Packet",
        "",
        f"- Trial rows inspected: `{packet['n_trials']}`",
        f"- Audit cases selected: `{packet['n_cases']}`",
        f"- Diagnostic cases per task: `{packet['cases_per_task']}`",
        f"- Passing controls per task: `{packet['controls_per_task']}`",
        f"- Semantic threshold: `{packet['semantic_threshold']}`",
        f"- Raw output included: `{packet['include_raw']}`",
        "",
        "Reviewers should retrieve the source input and gold labels using `task` and "
        "`sample_id`, then complete the blank human-review fields in the JSON or CSV file.",
        "",
    ]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in packet["cases"]:
        by_task[str(case["task"])].append(case)

    for task in sorted(by_task):
        lines.extend([f"## {task}", ""])
        for case in by_task[task]:
            lines.extend(
                [
                    f"### {case['audit_case_id']}",
                    "",
                    f"- Case type: `{case['case_type']}`",
                    f"- Review focus: {case['review_focus']}",
                    f"- Reasons: `{case['review_reasons']}`",
                    f"- Framework: `{case.get('framework', 'BLINDED')}`",
                    f"- Sample: `{case['sample_id']}`",
                    f"- Seed: `{case['seed']}`",
                    f"- Schema valid: `{case['schema_valid']}`",
                    f"- Quality: `{case['quality_score']}` (`{case['quality_metric']}`)",
                    f"- Low field scores: `{case['low_field_scores']}`",
                    "",
                    "Parsed output excerpt:",
                    "",
                    "```json",
                    case["parsed_output_excerpt"],
                    "```",
                    "",
                    "Reviewer fields: `human_correct`, `human_supported`, "
                    "`human_error_category`, `human_severity`, `needs_adjudication`, "
                    "`reviewer_id`, `reviewer_notes`.",
                    "",
                ]
            )
            if case.get("raw_output_excerpt"):
                lines.extend(
                    ["Raw output excerpt:", "", "```text", case["raw_output_excerpt"], "```", ""]
                )
    return "\n".join(lines)


def write_packet(
    trials_path: Path,
    *,
    json_output: Path | None = None,
    csv_output: Path | None = None,
    markdown_output: Path | None = None,
    cases_per_task: int = 5,
    controls_per_task: int = 1,
    semantic_threshold: float = 0.99,
    include_raw: bool = False,
    max_chars: int = 1200,
) -> dict[str, Any]:
    packet = select_audit_cases(
        load_trials(trials_path),
        cases_per_task=cases_per_task,
        controls_per_task=controls_per_task,
        semantic_threshold=semantic_threshold,
        include_raw=include_raw,
        max_chars=max_chars,
    )
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(packet, indent=2) + "\n")
    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(packet["cases"][0]) if packet["cases"] else []
        with csv_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for case in packet["cases"]:
                writer.writerow({key: csv_value(value) for key, value in case.items()})
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(packet))
    return packet


def normalize_review_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def load_review_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            return list(data["cases"])
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON audit format: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported audit review file extension: {path.suffix}")


def completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completion_fields = [
        "human_correct",
        "human_supported",
        "human_error_category",
        "human_severity",
        "needs_adjudication",
    ]
    return [
        row
        for row in rows
        if any(normalize_review_value(row.get(field)) for field in completion_fields)
    ]


def counts_for(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = normalize_review_value(row.get(field))
        if value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def yes_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [normalize_review_value(row.get(field)) for row in rows]
    values = [value for value in values if value]
    if not values:
        return None
    yes_values = {"1", "true", "yes", "y", "correct", "supported"}
    return round(sum(1 for value in values if value in yes_values) / len(values), 4)


def pairwise_agreement(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = normalize_review_value(row.get(field))
        if value:
            grouped[str(row.get("audit_case_id", ""))].append(row)

    n_pairs = 0
    agreements = 0
    left_counts: dict[str, int] = defaultdict(int)
    right_counts: dict[str, int] = defaultdict(int)
    disagreements: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(grouped.items()):
        for left, right in combinations(case_rows, 2):
            left_reviewer = normalize_review_value(left.get("reviewer_id"))
            right_reviewer = normalize_review_value(right.get("reviewer_id"))
            if left_reviewer and right_reviewer and left_reviewer == right_reviewer:
                continue
            n_pairs += 1
            left_value = normalize_review_value(left.get(field))
            right_value = normalize_review_value(right.get(field))
            left_counts[left_value] += 1
            right_counts[right_value] += 1
            if left_value == right_value:
                agreements += 1
            else:
                disagreements.append(
                    {
                        "audit_case_id": case_id,
                        "left_reviewer": left.get("reviewer_id"),
                        "right_reviewer": right.get("reviewer_id"),
                        "left_value": left_value,
                        "right_value": right_value,
                    }
                )

    agreement_rate = round(agreements / n_pairs, 4) if n_pairs else None
    chance_agreement = None
    cohen_kappa = None
    if n_pairs:
        categories = set(left_counts) | set(right_counts)
        chance = sum(
            (left_counts[category] / n_pairs) * (right_counts[category] / n_pairs)
            for category in categories
        )
        chance_agreement = round(chance, 4)
        if chance < 1.0:
            cohen_kappa = round(((agreements / n_pairs) - chance) / (1.0 - chance), 4)

    return {
        "field": field,
        "n_pairs": n_pairs,
        "agreements": agreements,
        "agreement_rate": agreement_rate,
        "chance_agreement": chance_agreement,
        "cohen_kappa": cohen_kappa,
        "disagreements": disagreements,
    }


def summarize_reviews(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = completed_rows(rows)
    unique_cases = {str(row.get("audit_case_id", "")) for row in rows}
    unique_completed = {str(row.get("audit_case_id", "")) for row in completed}
    return {
        "n_rows": len(rows),
        "n_unique_cases": len(unique_cases),
        "n_completed_rows": len(completed),
        "n_unique_completed_cases": len(unique_completed),
        "completion_rate": round(len(unique_completed) / len(unique_cases), 4)
        if unique_cases
        else 0.0,
        "task_counts": counts_for(rows, "task"),
        "case_type_counts": counts_for(rows, "case_type"),
        "reviewer_counts": counts_for(completed, "reviewer_id"),
        "human_correct_counts": counts_for(completed, "human_correct"),
        "human_correct_yes_rate": yes_rate(completed, "human_correct"),
        "human_supported_counts": counts_for(completed, "human_supported"),
        "human_supported_yes_rate": yes_rate(completed, "human_supported"),
        "needs_adjudication_counts": counts_for(completed, "needs_adjudication"),
        "error_category_counts": counts_for(completed, "human_error_category"),
        "severity_counts": counts_for(completed, "human_severity"),
        "agreement": {
            "human_correct": pairwise_agreement(completed, "human_correct"),
            "human_supported": pairwise_agreement(completed, "human_supported"),
        },
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Human Audit Summary",
        "",
        f"- Review rows: `{summary['n_rows']}`",
        f"- Unique audit cases: `{summary['n_unique_cases']}`",
        f"- Completed review rows: `{summary['n_completed_rows']}`",
        f"- Completed unique cases: `{summary['n_unique_completed_cases']}`",
        f"- Unique-case completion rate: `{summary['completion_rate']}`",
        f"- Human-correct yes rate: `{summary['human_correct_yes_rate']}`",
        f"- Human-supported yes rate: `{summary['human_supported_yes_rate']}`",
        "",
        "## Counts",
        "",
        f"- Tasks: `{summary['task_counts']}`",
        f"- Case types: `{summary['case_type_counts']}`",
        f"- Reviewers: `{summary['reviewer_counts']}`",
        f"- Error categories: `{summary['error_category_counts']}`",
        f"- Severity: `{summary['severity_counts']}`",
        f"- Needs adjudication: `{summary['needs_adjudication_counts']}`",
        "",
        "## Pairwise Agreement",
        "",
    ]
    for field, agreement in summary["agreement"].items():
        lines.extend(
            [
                f"### {field}",
                "",
                f"- Review pairs: `{agreement['n_pairs']}`",
                f"- Agreements: `{agreement['agreements']}`",
                f"- Agreement rate: `{agreement['agreement_rate']}`",
                f"- Chance agreement: `{agreement['chance_agreement']}`",
                f"- Cohen kappa: `{agreement['cohen_kappa']}`",
                "",
            ]
        )
        if agreement["disagreements"]:
            lines.append("Disagreements:")
            for item in agreement["disagreements"][:10]:
                lines.append(
                    "- "
                    f"`{item['audit_case_id']}`: "
                    f"`{item['left_reviewer']}`={item['left_value']}, "
                    f"`{item['right_reviewer']}`={item['right_value']}"
                )
            lines.append("")
    return "\n".join(lines)


def write_summary(
    review_path: Path,
    *,
    summary_output: Path | None = None,
    markdown_output: Path | None = None,
) -> dict[str, Any]:
    summary = summarize_reviews(load_review_rows(review_path))
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_summary_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a human-review audit packet")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--summary-markdown-output", type=Path, default=None)
    parser.add_argument("--cases-per-task", type=int, default=5)
    parser.add_argument("--controls-per-task", type=int, default=1)
    parser.add_argument("--semantic-threshold", type=float, default=0.99)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--max-chars", type=int, default=1200)
    args = parser.parse_args()

    if args.summarize:
        summary = write_summary(
            args.input_path,
            summary_output=args.summary_output,
            markdown_output=args.summary_markdown_output,
        )
        if not args.summary_output and not args.summary_markdown_output:
            print(render_summary_markdown(summary))
        else:
            if args.summary_output:
                print(f"Wrote {args.summary_output}")
            if args.summary_markdown_output:
                print(f"Wrote {args.summary_markdown_output}")
        return

    packet = write_packet(
        args.input_path,
        json_output=args.json_output,
        csv_output=args.csv_output,
        markdown_output=args.markdown_output,
        cases_per_task=args.cases_per_task,
        controls_per_task=args.controls_per_task,
        semantic_threshold=args.semantic_threshold,
        include_raw=args.include_raw,
        max_chars=args.max_chars,
    )
    if not args.json_output and not args.csv_output and not args.markdown_output:
        print(render_markdown(packet))
    else:
        if args.json_output:
            print(f"Wrote {args.json_output}")
        if args.csv_output:
            print(f"Wrote {args.csv_output}")
        if args.markdown_output:
            print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
