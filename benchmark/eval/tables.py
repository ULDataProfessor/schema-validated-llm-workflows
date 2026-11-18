"""Validate benchmark result directories and render LaTeX result tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

FRAMEWORK_LABELS = {
    "pydantic_ai": "PydanticAI",
    "plain_sdk": "Plain SDK",
    "langgraph": "LangGraph",
    "dspy": "DSPy",
    "crewai": "CrewAI",
    "agno": "Agno",
    "dry_run": "Dry run",
}

TASK_LABELS = {
    "extraction": "Structured extraction",
    "sql_generation": "SQL generation",
    "summarization": "Business report summarization",
    "compliance_qa": "Compliance QA",
    "multi_step": "Multi-step workflow",
}


@dataclass
class ValidationReport:
    eligible: bool
    problems: list[str]
    warnings: list[str]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_trial_config(config_path: Path = ROOT / "config" / "trials.yaml") -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text())


def expected_trials_per_group(meta: dict[str, Any], task: str, config: dict[str, Any]) -> int:
    seeds = meta.get("seeds") or ([meta["seed"]] if meta.get("seed") is not None else [])
    task_limits = meta.get("task_limits") or config.get("sample_limits", {})
    limit = task_limits.get(task) or config.get("sample_limits", {}).get(task)
    return int(limit or 0) * len(seeds)


def validate_result_dir(
    result_dir: Path,
    *,
    config_path: Path = ROOT / "config" / "trials.yaml",
    allow_dry_run: bool = False,
    allow_pilot: bool = False,
) -> ValidationReport:
    config = load_trial_config(config_path)
    problems: list[str] = []
    warnings: list[str] = []

    meta_path = result_dir / "meta.json"
    trials_path = result_dir / "trials.jsonl"
    summary_path = result_dir / "summary.json"
    for path in (meta_path, trials_path, summary_path):
        if not path.exists():
            problems.append(f"Missing required file: {path.name}")
    if problems:
        return ValidationReport(False, problems, warnings)

    meta = load_json(meta_path)
    summary = load_json(summary_path)
    dry_run = bool(meta.get("dry_run"))
    pilot = bool(meta.get("pilot"))
    if dry_run and not allow_dry_run:
        problems.append("Run is dry_run=true; dry runs are not eligible for empirical tables")
    if pilot and not allow_pilot:
        problems.append("Run is pilot=true; pilot runs are not eligible for final empirical tables")
    if pilot and allow_pilot:
        warnings.append("Run is pilot=true; tables are for smoke/pilot reporting only")

    expected_frameworks = ["dry_run"] if dry_run else config["frameworks"]
    expected_tasks = config["tasks"]
    observed_frameworks = meta.get("frameworks", [])
    observed_tasks = meta.get("tasks", [])
    if observed_frameworks != expected_frameworks:
        problems.append(
            f"Framework list mismatch: expected {expected_frameworks}, observed {observed_frameworks}"
        )
    if observed_tasks != expected_tasks:
        problems.append(f"Task list mismatch: expected {expected_tasks}, observed {observed_tasks}")

    expected_seeds = config.get("seeds", [])
    observed_seeds = meta.get("seeds") or ([meta["seed"]] if meta.get("seed") is not None else [])
    if not dry_run and not (pilot and allow_pilot) and observed_seeds != expected_seeds:
        problems.append(f"Seed list mismatch: expected {expected_seeds}, observed {observed_seeds}")

    expected_keys = {
        f"{framework}::{task}" for framework in expected_frameworks for task in expected_tasks
    }
    observed_keys = set(summary)
    missing_keys = sorted(expected_keys - observed_keys)
    extra_keys = sorted(observed_keys - expected_keys)
    if missing_keys:
        problems.append(f"Missing framework-task summaries: {missing_keys}")
    if extra_keys:
        warnings.append(f"Unexpected framework-task summaries: {extra_keys}")

    for key in sorted(expected_keys & observed_keys):
        _, task = key.split("::", 1)
        expected_n = expected_trials_per_group(meta, task, config)
        observed_n = int(summary[key].get("n", 0))
        if expected_n and observed_n != expected_n:
            problems.append(f"{key} has n={observed_n}; expected n={expected_n}")

    return ValidationReport(not problems, problems, warnings)


def fmt(value: Any, *, digits: int = 2, percent: bool = False) -> str:
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    if percent:
        return f"{float(value) * 100:.1f}"
    return f"{float(value):.{digits}f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def render_task_table(
    summary: dict[str, Any],
    *,
    task: str,
    frameworks: list[str],
) -> str:
    label = TASK_LABELS.get(task, task.replace("_", " ").title())
    metric_names = sorted(
        {
            str(stats.get("quality_metric"))
            for framework in frameworks
            if (stats := summary.get(f"{framework}::{task}")) and stats.get("quality_metric")
        }
    )
    metric_note = ""
    if len(metric_names) == 1:
        metric_note = f" Quality metric: \\texttt{{{latex_escape(metric_names[0])}}}."
    elif len(metric_names) > 1:
        metric_note = (
            " Quality metrics: "
            + ", ".join(f"\\texttt{{{latex_escape(name)}}}" for name in metric_names)
            + "."
        )
    rows = []
    for framework in frameworks:
        stats = summary.get(f"{framework}::{task}")
        framework_label = FRAMEWORK_LABELS.get(framework, framework)
        if not stats:
            rows.append(f"{latex_escape(framework_label)} & -- & -- & -- & -- & -- & -- & missing \\\\")
            continue
        notes = []
        if stats.get("error_count", 0):
            notes.append(f"errors={stats['error_count']}")
        row = (
            f"{latex_escape(framework_label)} & "
            f"{int(stats['n'])} & "
            f"{fmt(stats.get('zero_retry_success_rate'), percent=True)} & "
            f"{fmt(stats.get('schema_pass_rate'), percent=True)} & "
            f"{fmt(stats.get('mean_quality_score'))} & "
            f"{fmt(stats.get('mean_retries'))} & "
            f"{fmt(stats.get('p95_latency_ms'), digits=1)} & "
            f"{latex_escape('; '.join(notes) if notes else '')} \\\\"
        )
        rows.append(row)

    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabularx}}{{\\textwidth}}{{p{{0.16\\textwidth}}rrrrrrX}}
\\toprule
Framework & $n$ & ZRS & FSV & Quality & Retries & p95 ms & Notes \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
\\caption{{Generated primary result table for {latex_escape(label)}. ZRS and FSV are percentages.{metric_note}}}
\\label{{tab:generated-{task.replace('_', '-')}}}
\\end{{table}}
"""


def render_all_tables(
    result_dir: Path,
    *,
    allow_dry_run: bool = False,
    allow_pilot: bool = False,
) -> str:
    return render_all_tables_with_config(
        result_dir,
        config_path=ROOT / "config" / "trials.yaml",
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )


def render_all_tables_with_config(
    result_dir: Path,
    *,
    config_path: Path,
    allow_dry_run: bool = False,
    allow_pilot: bool = False,
) -> str:
    meta = load_json(result_dir / "meta.json")
    summary = load_json(result_dir / "summary.json")
    report = validate_result_dir(
        result_dir,
        config_path=config_path,
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )
    frameworks = meta.get("frameworks", [])
    tasks = meta.get("tasks", [])
    if report.eligible and meta.get("pilot"):
        eligibility = "pilot-only"
    elif report.eligible:
        eligibility = "eligible"
    else:
        eligibility = "not eligible"

    comments = [
        "% Generated by python -m eval.tables.",
        f"% Result directory: {result_dir}",
        f"% Eligibility: {eligibility}",
    ]
    comments.extend(f"% PROBLEM: {problem}" for problem in report.problems)
    comments.extend(f"% WARNING: {warning}" for warning in report.warnings)
    tables = [render_task_table(summary, task=task, frameworks=frameworks) for task in tasks]
    return "\n".join(comments + [""] + tables)


def write_tables(
    result_dir: Path,
    output_path: Path,
    *,
    config_path: Path = ROOT / "config" / "trials.yaml",
    allow_dry_run: bool = False,
    allow_pilot: bool = False,
) -> ValidationReport:
    rendered = render_all_tables_with_config(
        result_dir,
        config_path=config_path,
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    return validate_result_dir(
        result_dir,
        config_path=config_path,
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render LaTeX tables from benchmark summaries")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "trials.yaml")
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if not report-eligible")
    args = parser.parse_args()

    output_path = args.output or args.result_dir / "tables.tex"
    report = write_tables(
        args.result_dir,
        output_path,
        config_path=args.config,
        allow_dry_run=args.allow_dry_run,
        allow_pilot=args.allow_pilot,
    )
    print(f"Wrote {output_path}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for problem in report.problems:
        print(f"PROBLEM: {problem}")
    if args.strict and not report.eligible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
