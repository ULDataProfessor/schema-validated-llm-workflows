"""Select qualitative failure examples from benchmark trial records."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.report import classify_error, load_trials


CATEGORY_LABELS = {
    "parse_or_transport_failure": "Parse or transport failure",
    "schema_violation": "Schema violation",
    "schema_valid_semantic_error": "Schema-valid semantic error",
    "tool_or_orchestration_error": "Tool or orchestration error",
    "state_or_procedure_failure": "State or procedure failure",
    "retry_exhaustion": "Retry exhaustion",
    "recovered_retry": "Recovered retry",
}

TOOL_FIELDS = {
    "request_classification",
    "tool_used",
    "tool_selection",
    "workflow_route",
}

STATE_PROCEDURE_FIELDS = {
    "requires_human_review",
    "final_disposition",
    "task_completion",
    "state_consistency",
    "procedure_compliance",
}

SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer|token|password)\b\s*[:=]\s*['\"]?[^'\"\s,}]+"
)
BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]+")


def redact(text: str) -> str:
    text = BEARER_PATTERN.sub("Bearer <redacted>", text)
    return SECRET_PATTERN.sub(r"\1=<redacted>", text)


def excerpt(value: Any, *, max_chars: int = 800) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, default=str)
    text = redact(text.strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def quality_score(trial: dict[str, Any]) -> float | None:
    if trial.get("quality_score") is not None:
        return float(trial["quality_score"])
    if trial.get("field_f1") is not None:
        return float(trial["field_f1"])
    return None


def low_field_scores(
    trial: dict[str, Any],
    *,
    threshold: float = 1.0,
) -> dict[str, float]:
    scores = trial.get("field_scores") or {}
    return {
        str(field): float(score)
        for field, score in sorted(scores.items())
        if float(score) < threshold
    }


def failure_categories(
    trial: dict[str, Any],
    *,
    semantic_threshold: float = 0.99,
) -> list[str]:
    categories: list[str] = []
    error_type = classify_error(trial.get("error"))
    schema_valid = bool(trial.get("schema_valid"))
    retries = int(trial.get("retries", 0) or 0)
    quality = quality_score(trial)
    fields = low_field_scores(trial)

    if error_type in {"parse_error", "provider_request", "authentication", "missing_dependency"}:
        categories.append("parse_or_transport_failure")
    elif error_type == "schema_violation":
        categories.append("schema_violation")
    elif error_type == "retry_exhaustion":
        categories.append("retry_exhaustion")
    elif error_type:
        categories.append("parse_or_transport_failure")

    if not schema_valid and "schema_violation" not in categories:
        categories.append("schema_violation")

    if schema_valid and quality is not None and quality < semantic_threshold:
        categories.append("schema_valid_semantic_error")

    if trial.get("task") == "multi_step" and any(field in TOOL_FIELDS for field in fields):
        categories.append("tool_or_orchestration_error")

    if trial.get("task") == "multi_step" and any(
        field in STATE_PROCEDURE_FIELDS for field in fields
    ):
        categories.append("state_or_procedure_failure")

    if retries > 0 and not schema_valid and "retry_exhaustion" not in categories:
        categories.append("retry_exhaustion")
    elif retries > 0 and schema_valid:
        categories.append("recovered_retry")

    return list(dict.fromkeys(categories))


def severity_key(trial: dict[str, Any], category: str) -> tuple:
    quality = quality_score(trial)
    quality_value = quality if quality is not None else 1.0
    retries = int(trial.get("retries", 0) or 0)
    latency = float(trial.get("latency_ms", 0) or 0)
    base = (
        str(trial.get("task", "")),
        str(trial.get("framework", "")),
        str(trial.get("sample_id", "")),
        str(trial.get("seed", "")),
    )
    if category in {
        "schema_valid_semantic_error",
        "tool_or_orchestration_error",
        "state_or_procedure_failure",
    }:
        return (quality_value, -retries, -latency, *base)
    if category in {"retry_exhaustion", "recovered_retry"}:
        return (-retries, quality_value, -latency, *base)
    return (0 if trial.get("error") else 1, quality_value, -retries, *base)


def summarize_trial(
    trial: dict[str, Any],
    *,
    category: str,
    max_chars: int = 800,
    include_raw: bool = False,
) -> dict[str, Any]:
    summary = {
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "framework": trial.get("framework"),
        "task": trial.get("task"),
        "sample_id": trial.get("sample_id"),
        "seed": trial.get("seed"),
        "schema_valid": bool(trial.get("schema_valid")),
        "quality_metric": trial.get("quality_metric"),
        "quality_score": quality_score(trial),
        "retries": int(trial.get("retries", 0) or 0),
        "latency_ms": trial.get("latency_ms"),
        "tokens_total": (trial.get("tokens_in", 0) or 0) + (trial.get("tokens_out", 0) or 0),
        "error_type": classify_error(trial.get("error")),
        "error_excerpt": excerpt(trial.get("error"), max_chars=max_chars),
        "low_field_scores": low_field_scores(trial),
        "parsed_output_excerpt": excerpt(trial.get("parsed_output"), max_chars=max_chars),
    }
    if include_raw:
        summary["raw_output_excerpt"] = excerpt(trial.get("raw_output"), max_chars=max_chars)
    return summary


def select_examples(
    trials: list[dict[str, Any]],
    *,
    per_category: int = 3,
    semantic_threshold: float = 0.99,
    include_raw: bool = False,
    max_chars: int = 800,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        for category in failure_categories(trial, semantic_threshold=semantic_threshold):
            grouped[category].append(trial)

    examples: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for category, category_trials in sorted(grouped.items()):
        ordered = sorted(category_trials, key=lambda trial: severity_key(trial, category))
        counts[category] = len(ordered)
        examples[category] = [
            summarize_trial(
                trial,
                category=category,
                max_chars=max_chars,
                include_raw=include_raw,
            )
            for trial in ordered[:per_category]
        ]

    return {
        "n_trials": len(trials),
        "per_category": per_category,
        "semantic_threshold": semantic_threshold,
        "include_raw": include_raw,
        "category_counts": counts,
        "examples": examples,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qualitative Failure Examples",
        "",
        f"- Trial rows inspected: `{report['n_trials']}`",
        f"- Examples per category: `{report['per_category']}`",
        f"- Semantic threshold: `{report['semantic_threshold']}`",
        f"- Raw output included: `{report['include_raw']}`",
        "",
    ]
    if not report["examples"]:
        lines.extend(["No failures or diagnostic examples matched the configured criteria.", ""])
        return "\n".join(lines)

    for category, examples in report["examples"].items():
        label = CATEGORY_LABELS.get(category, category)
        count = report["category_counts"].get(category, 0)
        lines.extend([f"## {label}", "", f"Matched trial rows: `{count}`", ""])
        for index, example in enumerate(examples, start=1):
            lines.extend(
                [
                    f"### Example {index}",
                    "",
                    f"- Framework: `{example['framework']}`",
                    f"- Task: `{example['task']}`",
                    f"- Sample: `{example['sample_id']}`",
                    f"- Seed: `{example['seed']}`",
                    f"- Schema valid: `{example['schema_valid']}`",
                    f"- Quality: `{example['quality_score']}` "
                    f"(`{example['quality_metric']}`)",
                    f"- Retries: `{example['retries']}`",
                    f"- Error type: `{example['error_type']}`",
                    f"- Low field scores: `{example['low_field_scores']}`",
                    "",
                ]
            )
            if example.get("error_excerpt"):
                lines.extend(["Error excerpt:", "", "```text", example["error_excerpt"], "```", ""])
            if example.get("parsed_output_excerpt"):
                lines.extend(
                    [
                        "Parsed output excerpt:",
                        "",
                        "```json",
                        example["parsed_output_excerpt"],
                        "```",
                        "",
                    ]
                )
            if example.get("raw_output_excerpt"):
                lines.extend(["Raw output excerpt:", "", "```text", example["raw_output_excerpt"], "```", ""])
    return "\n".join(lines)


def write_examples(
    trials_path: Path,
    *,
    markdown_output: Path | None = None,
    json_output: Path | None = None,
    per_category: int = 3,
    semantic_threshold: float = 0.99,
    include_raw: bool = False,
    max_chars: int = 800,
) -> dict[str, Any]:
    trials = load_trials(trials_path)
    report = select_examples(
        trials,
        per_category=per_category,
        semantic_threshold=semantic_threshold,
        include_raw=include_raw,
        max_chars=max_chars,
    )
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2) + "\n")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select qualitative failure examples")
    parser.add_argument("trials_path", type=Path)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--semantic-threshold", type=float, default=0.99)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--max-chars", type=int, default=800)
    args = parser.parse_args()

    report = write_examples(
        args.trials_path,
        markdown_output=args.markdown_output,
        json_output=args.json_output,
        per_category=args.per_category,
        semantic_threshold=args.semantic_threshold,
        include_raw=args.include_raw,
        max_chars=args.max_chars,
    )
    if not args.markdown_output and not args.json_output:
        print(render_markdown(report))
    else:
        if args.markdown_output:
            print(f"Wrote {args.markdown_output}")
        if args.json_output:
            print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
