"""Render a transparent benchmark result card for one run directory."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

from eval.tables import (
    FRAMEWORK_LABELS,
    ROOT,
    TASK_LABELS,
    ValidationReport,
    load_json,
    validate_result_dir,
)


def markdown_list(values: list[Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def render_command(argv: list[Any]) -> str:
    if not argv:
        return "unknown"
    return " ".join(shlex.quote(str(part)) for part in argv)


def percentage(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value) * 100:.1f}%"


def number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def eligibility_label(meta: dict[str, Any], report: ValidationReport) -> str:
    if not report.eligible:
        return "not eligible"
    if meta.get("pilot"):
        return "pilot-only"
    if meta.get("dry_run"):
        return "dry-run only"
    return "eligible"


def claim_scope(meta: dict[str, Any], report: ValidationReport) -> str:
    if not report.eligible:
        return "debug or incomplete run; exclude from empirical claims"
    if meta.get("dry_run"):
        return "software-path smoke evidence; no live model reliability claim"
    if meta.get("pilot"):
        return "runner-provider smoke evidence; no framework ranking claim"
    return "descriptive empirical comparison; inferential claims require paired analysis"


def claim_evidence_checklist(meta: dict[str, Any], report: ValidationReport) -> list[str]:
    """Return claim-specific evidence notes for the generated result card."""
    if not report.eligible:
        return [
            "This run is incomplete or inconsistent with the configured benchmark.",
            "Use it for debugging only; exclude it from empirical tables and framework claims.",
            "Regenerate the result card after the run satisfies the configured matrix.",
        ]
    if meta.get("dry_run"):
        return [
            "Dry-run rows can support software-path claims only.",
            "Live model reliability, latency, token cost, and framework ranking remain unsupported.",
            "A provider-backed pilot is required before runner-provider readiness claims.",
        ]
    if meta.get("pilot"):
        return [
            "Pilot rows can support runner-provider readiness claims only.",
            "Framework ranking requires the complete configured matrix, not one-seed pilot rows.",
            "Inferential claims also require paired-analysis files and failure-example review.",
        ]
    if report.eligible:
        return [
            "This run can support descriptive framework-task comparisons.",
            "Inferential claims require paired-analysis JSON files on shared sample/seed keys.",
            "Cost claims require generated cost-analysis artifacts with explicit price assumptions.",
            "Failure-mode claims require deterministic failure-example artifacts.",
            "Semantic audit claims require human-audit packets, completed review fields, and summary files.",
        ]
    raise AssertionError("unreachable result-card claim evidence branch")


def render_metric_table(
    summary: dict[str, Any],
    *,
    frameworks: list[str],
    tasks: list[str],
) -> str:
    rows = [
        "| Task | Framework | n | ZRS | FSV | Quality | Quality metric | Errors |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for task in tasks:
        task_label = TASK_LABELS.get(task, task)
        for framework in frameworks:
            framework_label = FRAMEWORK_LABELS.get(framework, framework)
            stats = summary.get(f"{framework}::{task}", {})
            rows.append(
                "| "
                + " | ".join(
                    [
                        task_label,
                        framework_label,
                        str(stats.get("n", "--")),
                        percentage(stats.get("zero_retry_success_rate")),
                        percentage(stats.get("schema_pass_rate")),
                        number(stats.get("mean_quality_score")),
                        str(stats.get("quality_metric", "--")),
                        str(stats.get("error_count", "--")),
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def validation_section(report: ValidationReport) -> str:
    lines = ["## Validation"]
    if report.problems:
        lines.append("")
        lines.append("Problems:")
        lines.extend(f"- {problem}" for problem in report.problems)
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report.warnings)
    if not report.problems and not report.warnings:
        lines.append("")
        lines.append("No validation problems or warnings were reported.")
    return "\n".join(lines)


def render_result_card(
    result_dir: Path,
    *,
    config_path: Path = ROOT / "config" / "trials.yaml",
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
    frameworks = list(meta.get("frameworks", []))
    tasks = list(meta.get("tasks", []))
    seeds = meta.get("seeds") or ([meta["seed"]] if meta.get("seed") is not None else [])
    task_limits = meta.get("task_limits") or {}
    environment = meta.get("environment") or {}
    provenance = meta.get("provenance") or {}
    package_versions = environment.get("package_versions") or {}
    git_info = environment.get("git") or {}
    config_hashes = environment.get("config_hashes") or {}

    lines = [
        f"# Benchmark Result Card: {meta.get('run_id', result_dir.name)}",
        "",
        "## Scope",
        "",
        f"- Eligibility: **{eligibility_label(meta, report)}**",
        f"- Claim scope: {claim_scope(meta, report)}",
        f"- Result directory: `{result_dir}`",
        f"- Model: `{meta.get('model', 'unknown')}`",
        f"- Started at: `{provenance.get('started_at_utc', 'unknown')}`",
        f"- Completed at: `{provenance.get('completed_at_utc', 'unknown')}`",
        f"- Command: `{render_command(provenance.get('command_argv') or [])}`",
        f"- Python: `{environment.get('python_version', 'unknown')}`",
        f"- Platform: `{environment.get('platform', 'unknown')}`",
        f"- Git commit: `{git_info.get('commit', 'unknown')}`",
        f"- Git dirty: `{git_info.get('dirty', 'unknown')}`",
        f"- Dry run: `{bool(meta.get('dry_run'))}`",
        f"- Pilot: `{bool(meta.get('pilot'))}`",
        f"- Frameworks: {markdown_list(frameworks)}",
        f"- Tasks: {markdown_list(tasks)}",
        f"- Seeds: {markdown_list(seeds)}",
        f"- Task limits: `{task_limits}`",
        f"- Planned trials: `{meta.get('planned_trials', 'unknown')}`",
        f"- Recorded trials: `{meta.get('n_trials', 'unknown')}`",
        f"- Package versions: `{package_versions}`",
        f"- Config hashes: `{config_hashes}`",
        "",
        "## Primary Metrics",
        "",
        render_metric_table(summary, frameworks=frameworks, tasks=tasks),
        "",
        validation_section(report),
        "",
        "## Claim Evidence Checklist",
        "",
    ]
    lines.extend(f"- {item}" for item in claim_evidence_checklist(meta, report))
    lines.extend(
        [
            "",
            "## Required Artifacts",
            "",
            "- `meta.json` records run settings, framework/task lists, seeds, status, command provenance, environment, package versions, and config hashes.",
            "- `trials.jsonl` is the authoritative row-level evidence for recomputation.",
            "- `summary.json` aggregates metrics derived from `trials.jsonl`.",
            "- `tables.tex` is generated from `summary.json` after eligibility checks.",
            "- `failure_examples.md` and `failure_examples.json` support qualitative review.",
            "- `human_audit.md`, `human_audit.csv`, and `human_audit.json` support reviewer calibration.",
            "- `human_audit_summary.md` and `human_audit_summary.json` summarize completed reviews.",
            "- `judge_triage.md` and `judge_triage.json` support review prioritization when judge labels are used.",
            "- `cost_analysis.md` and `cost_analysis.json` document token-cost assumptions.",
            "- Paired-analysis JSON files are required for inferential framework comparisons.",
            "- `ablation_manifest_audit.json` documents pre-specified mechanism checks.",
            "- `stress_manifest.yaml` and `stress_manifest_audit.json` document stress-extension conditions.",
            "- `maintainability.csv` and `maintainability_summary.json` support schema-change effort claims.",
            "- `artifact_audit.md` and `artifact_audit.json` record executable claim-readiness checks.",
            "",
            "## Interpretation Limits",
            "",
            "- Dry-run and pilot cards document artifact readiness, not framework superiority.",
            "- Final empirical claims require the configured full matrix and archived trial rows.",
            "- Statistical claims should be backed by paired comparisons on shared samples.",
            "- Raw text may require redaction or hashing before public release.",
            "",
        ]
    )
    return "\n".join(lines)


def write_result_card(
    result_dir: Path,
    output_path: Path,
    *,
    config_path: Path = ROOT / "config" / "trials.yaml",
    allow_dry_run: bool = False,
    allow_pilot: bool = False,
) -> ValidationReport:
    card = render_result_card(
        result_dir,
        config_path=config_path,
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card)
    return validate_result_dir(
        result_dir,
        config_path=config_path,
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a benchmark result card")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "trials.yaml")
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if not report-eligible")
    args = parser.parse_args()

    output_path = args.output or args.result_dir / "result_card.md"
    report = write_result_card(
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
