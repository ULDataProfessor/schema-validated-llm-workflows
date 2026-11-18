"""Audit benchmark result directories against claim-specific evidence tiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.tables import ROOT, ValidationReport, load_json, validate_result_dir


ARTIFACT_GROUPS = {
    "core": (
        "meta.json",
        "trials.jsonl",
        "summary.json",
        "tables.tex",
        "result_card.md",
    ),
    "failure_analysis": (
        "failure_examples.md",
        "failure_examples.json",
    ),
    "human_audit_packet": (
        "human_audit.md",
        "human_audit.csv",
        "human_audit.json",
    ),
    "human_audit_summary": (
        "human_audit_summary.md",
        "human_audit_summary.json",
    ),
    "cost_analysis": (
        "cost_analysis.md",
        "cost_analysis.json",
    ),
    "ablation_manifest": (
        "ablation_manifest_audit.md",
        "ablation_manifest_audit.json",
    ),
    "stress_manifest": (
        "stress_manifest.yaml",
        "stress_manifest_audit.md",
        "stress_manifest_audit.json",
    ),
    "maintainability": (
        "maintainability.md",
        "maintainability.csv",
        "maintainability.json",
        "maintainability_summary.md",
        "maintainability_summary.json",
    ),
}

PAIRED_ANALYSIS_PATTERNS = ("paired_*.json", "*paired*.json")
OPTIONAL_ARTIFACT_PATTERNS = (
    "claim_records.md",
    "claim_records.json",
    "judge_triage.md",
    "judge_triage.json",
    "stress_trials.jsonl",
    "stress_summary.json",
    "stress_paired_*.json",
)

CLAIM_LABELS = {
    "software_path": "Software-path smoke evidence",
    "runner_readiness": "Live runner readiness",
    "descriptive_results": "Descriptive framework-task results",
    "inferential_claims": "Inferential framework comparison",
    "failure_analysis": "Failure-mode analysis",
    "cost_analysis": "Cost analysis",
    "mechanism_ablation_manifest": "Mechanism ablation manifest",
    "stress_manifest": "Stress manifest",
    "human_audit_packet": "Human-audit packet",
    "semantic_human_audit": "Completed semantic human audit",
    "maintainability_review": "Maintainability review",
    "environment_metadata": "Environment metadata",
    "run_provenance": "Run provenance",
}


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


def validate_safely(
    result_dir: Path,
    *,
    config_path: Path,
    allow_dry_run: bool,
    allow_pilot: bool,
) -> ValidationReport:
    try:
        return validate_result_dir(
            result_dir,
            config_path=config_path,
            allow_dry_run=allow_dry_run,
            allow_pilot=allow_pilot,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        message = f"Validation failed before eligibility checks: {type(exc).__name__}: {exc}"
        return ValidationReport(False, [message], [])


def load_json_safely(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        data = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(data, dict):
        return data
    return {"_value": data}


def file_status(result_dir: Path, relative_path: str) -> dict[str, Any]:
    path = result_dir / relative_path
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": relative_path,
        "exists": exists,
        "nonempty": exists and size_bytes > 0,
        "size_bytes": size_bytes,
    }


def group_status(result_dir: Path, filenames: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {filename: file_status(result_dir, filename) for filename in filenames}


def missing_from(group: dict[str, dict[str, Any]]) -> list[str]:
    return [filename for filename, status in group.items() if not status["nonempty"]]


def glob_status(result_dir: Path, patterns: tuple[str, ...]) -> dict[str, Any]:
    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(result_dir.glob(pattern)):
            if path.is_file():
                files[path.name] = path

    statuses = {name: file_status(result_dir, name) for name in sorted(files)}
    nonempty = [name for name, status in statuses.items() if status["nonempty"]]
    return {
        "patterns": list(patterns),
        "files": statuses,
        "nonempty_files": nonempty,
        "count": len(nonempty),
    }


def environment_metadata_ready(meta: dict[str, Any]) -> bool:
    environment = meta.get("environment") or {}
    return bool(
        environment.get("python_version")
        and environment.get("platform")
        and environment.get("package_versions")
        and environment.get("config_hashes")
    )


def run_provenance_ready(meta: dict[str, Any]) -> bool:
    provenance = meta.get("provenance") or {}
    return bool(
        provenance.get("started_at_utc")
        and provenance.get("working_directory")
        and provenance.get("command_argv")
        and provenance.get("args")
    )


def completed_human_review_rows(result_dir: Path) -> int:
    summary = load_json_safely(result_dir / "human_audit_summary.json")
    if summary.get("_load_error"):
        return 0
    try:
        return int(summary.get("n_completed_rows", 0) or 0)
    except (TypeError, ValueError):
        return 0


def completed_maintainability_rows(result_dir: Path) -> int:
    summary = load_json_safely(result_dir / "maintainability_summary.json")
    if summary.get("_load_error"):
        return 0
    frameworks = summary.get("frameworks") or {}
    total = 0
    for stats in frameworks.values():
        if not isinstance(stats, dict):
            continue
        try:
            total += int(stats.get("n_completed", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def stress_manifest_audit_ok(result_dir: Path) -> bool:
    audit = load_json_safely(result_dir / "stress_manifest_audit.json")
    if audit.get("_load_error"):
        return False
    return audit.get("ok") is True


def audit_notes(
    *,
    meta: dict[str, Any],
    report: ValidationReport,
    missing_artifacts: dict[str, list[str]],
    paired_count: int,
    completed_reviews: int,
    completed_maintainability: int,
    stress_manifest_ok: bool,
) -> list[str]:
    notes: list[str] = []
    if report.problems:
        notes.append("Result validation reported problems; empirical claims are not ready.")
    if missing_artifacts["core"]:
        notes.append("Core generated artifacts are incomplete.")
    if meta.get("dry_run"):
        notes.append("Dry-run rows support software-path checks only.")
    if meta.get("pilot"):
        notes.append("Pilot rows support runner-readiness checks, not framework ranking.")
    if paired_count == 0:
        notes.append("No paired-analysis JSON files were found.")
    if completed_reviews == 0:
        notes.append("Human-audit summary has no completed review rows.")
    if missing_artifacts["maintainability"]:
        notes.append("Maintainability worksheet or summary artifacts are incomplete.")
    elif completed_maintainability == 0:
        notes.append("Maintainability summary has no completed change-exercise rows.")
    if missing_artifacts["ablation_manifest"]:
        notes.append("Ablation-manifest audit artifacts are incomplete.")
    if (
        len(missing_artifacts["stress_manifest"])
        != len(ARTIFACT_GROUPS["stress_manifest"])
        and missing_artifacts["stress_manifest"]
    ):
        notes.append("Stress-manifest artifacts are partially complete.")
    elif not missing_artifacts["stress_manifest"] and not stress_manifest_ok:
        notes.append("Stress-manifest audit did not report ok=true.")
    if not environment_metadata_ready(meta):
        notes.append("Environment/package/config metadata are missing or incomplete.")
    if not run_provenance_ready(meta):
        notes.append("Run command provenance is missing or incomplete.")
    return notes


def readiness_notes(
    *,
    readiness: dict[str, bool],
    missing_artifacts: dict[str, list[str]],
    paired_count: int,
    completed_reviews: int,
    completed_maintainability: int,
    stress_manifest_ok: bool,
) -> dict[str, str]:
    return {
        "software_path": "ready for dry-run software-path claims"
        if readiness["software_path"]
        else "requires an allowed dry run and complete core artifacts",
        "runner_readiness": "ready for provider-backed runner-readiness claims"
        if readiness["runner_readiness"]
        else "requires an allowed non-dry run and complete core artifacts",
        "descriptive_results": "ready for descriptive full-matrix tables"
        if readiness["descriptive_results"]
        else "requires a non-pilot, non-dry eligible run with complete core artifacts",
        "inferential_claims": f"{paired_count} paired-analysis file(s) available"
        if readiness["inferential_claims"]
        else "requires descriptive-results readiness plus paired-analysis JSON files",
        "failure_analysis": "failure-example Markdown and JSON files are available"
        if readiness["failure_analysis"]
        else f"missing {missing_artifacts['failure_analysis']}",
        "cost_analysis": "cost-analysis Markdown and JSON files are available"
        if readiness["cost_analysis"]
        else f"missing {missing_artifacts['cost_analysis']}",
        "mechanism_ablation_manifest": "ablation-manifest audit Markdown and JSON files are available"
        if readiness["mechanism_ablation_manifest"]
        else f"missing {missing_artifacts['ablation_manifest']}",
        "stress_manifest": "stress manifest and manifest-audit files are available"
        if readiness["stress_manifest"]
        else (
            f"missing {missing_artifacts['stress_manifest']}"
            if missing_artifacts["stress_manifest"]
            else f"requires stress_manifest_audit.json ok=true; current ok={stress_manifest_ok}"
        ),
        "human_audit_packet": "human-audit Markdown, CSV, and JSON packet files are available"
        if readiness["human_audit_packet"]
        else f"missing {missing_artifacts['human_audit_packet']}",
        "semantic_human_audit": f"{completed_reviews} completed human-review row(s) available"
        if readiness["semantic_human_audit"]
        else "requires audit packet files, summary files, and at least one completed review row",
        "maintainability_review": (
            f"{completed_maintainability} completed maintainability row(s) available"
        )
        if readiness["maintainability_review"]
        else "requires maintainability files and at least one completed change-exercise row",
        "environment_metadata": "environment snapshot, package versions, and config hashes are present"
        if readiness["environment_metadata"]
        else "requires environment.python_version, platform, package_versions, and config_hashes",
        "run_provenance": "run timestamps, working directory, command argv, and CLI args are present"
        if readiness["run_provenance"]
        else "requires provenance.started_at_utc, working_directory, command_argv, and args",
    }


def audit_result_dir(
    result_dir: Path,
    *,
    config_path: Path = ROOT / "config" / "trials.yaml",
    allow_dry_run: bool = False,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    report = validate_safely(
        result_dir,
        config_path=config_path,
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )
    meta = load_json_safely(result_dir / "meta.json")
    artifacts = {
        group_name: group_status(result_dir, filenames)
        for group_name, filenames in ARTIFACT_GROUPS.items()
    }
    missing_artifacts = {
        group_name: missing_from(statuses) for group_name, statuses in artifacts.items()
    }
    paired_analysis = glob_status(result_dir, PAIRED_ANALYSIS_PATTERNS)
    optional_artifacts = glob_status(result_dir, OPTIONAL_ARTIFACT_PATTERNS)

    dry_run = bool(meta.get("dry_run"))
    pilot = bool(meta.get("pilot"))
    core_complete = not missing_artifacts["core"]
    eligible_core = report.eligible and core_complete
    full_run = eligible_core and not dry_run and not pilot
    completed_reviews = completed_human_review_rows(result_dir)
    completed_maintainability = completed_maintainability_rows(result_dir)
    stress_manifest_ok = stress_manifest_audit_ok(result_dir)
    human_packet_ready = report.eligible and not missing_artifacts["human_audit_packet"]

    readiness = {
        "software_path": eligible_core and dry_run,
        "runner_readiness": eligible_core and not dry_run,
        "descriptive_results": full_run,
        "inferential_claims": full_run and paired_analysis["count"] > 0,
        "failure_analysis": report.eligible and not missing_artifacts["failure_analysis"],
        "cost_analysis": report.eligible and not missing_artifacts["cost_analysis"],
        "mechanism_ablation_manifest": (
            report.eligible and not missing_artifacts["ablation_manifest"]
        ),
        "stress_manifest": (
            report.eligible
            and not missing_artifacts["stress_manifest"]
            and stress_manifest_ok
        ),
        "human_audit_packet": human_packet_ready,
        "semantic_human_audit": (
            human_packet_ready
            and not missing_artifacts["human_audit_summary"]
            and completed_reviews > 0
        ),
        "maintainability_review": (
            report.eligible
            and not missing_artifacts["maintainability"]
            and completed_maintainability > 0
        ),
        "environment_metadata": environment_metadata_ready(meta),
        "run_provenance": run_provenance_ready(meta),
    }

    notes = readiness_notes(
        readiness=readiness,
        missing_artifacts=missing_artifacts,
        paired_count=paired_analysis["count"],
        completed_reviews=completed_reviews,
        completed_maintainability=completed_maintainability,
        stress_manifest_ok=stress_manifest_ok,
    )

    return {
        "result_dir": str(result_dir),
        "run_id": meta.get("run_id", result_dir.name),
        "eligibility": eligibility_label(meta, report),
        "claim_scope": claim_scope(meta, report),
        "strict_pass": eligible_core,
        "validation": {
            "eligible": report.eligible,
            "problems": report.problems,
            "warnings": report.warnings,
        },
        "run_flags": {
            "dry_run": dry_run,
            "pilot": pilot,
        },
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "paired_analysis": paired_analysis,
        "optional_artifacts": optional_artifacts,
        "human_audit": {
            "completed_review_rows": completed_reviews,
        },
        "maintainability": {
            "completed_rows": completed_maintainability,
        },
        "stress_manifest": {
            "audit_ok": stress_manifest_ok,
        },
        "readiness": readiness,
        "readiness_notes": notes,
        "notes": audit_notes(
            meta=meta,
            report=report,
            missing_artifacts=missing_artifacts,
            paired_count=paired_analysis["count"],
            completed_reviews=completed_reviews,
            completed_maintainability=completed_maintainability,
            stress_manifest_ok=stress_manifest_ok,
        ),
    }


def ready_text(value: bool) -> str:
    return "ready" if value else "not ready"


def render_artifact_audit(audit: dict[str, Any]) -> str:
    lines = [
        f"# Artifact Audit: {audit['run_id']}",
        "",
        "## Scope",
        "",
        f"- Result directory: `{audit['result_dir']}`",
        f"- Eligibility: **{audit['eligibility']}**",
        f"- Claim scope: {audit['claim_scope']}",
        f"- Strict core pass: `{audit['strict_pass']}`",
        f"- Dry run: `{audit['run_flags']['dry_run']}`",
        f"- Pilot: `{audit['run_flags']['pilot']}`",
        "",
        "## Claim Readiness",
        "",
        "| Claim tier | Status | Evidence note |",
        "|---|---|---|",
    ]
    for key, label in CLAIM_LABELS.items():
        lines.append(
            f"| {label} | {ready_text(audit['readiness'][key])} | "
            f"{audit['readiness_notes'][key]} |"
        )

    lines.extend(["", "## Artifacts", ""])
    for group_name, group in audit["artifacts"].items():
        lines.extend([f"### {group_name}", ""])
        for filename, status in group.items():
            state = "present" if status["nonempty"] else "missing"
            lines.append(f"- `{filename}`: {state} ({status['size_bytes']} bytes)")
        lines.append("")

    paired = audit["paired_analysis"]
    lines.extend(
        [
            "### paired_analysis",
            "",
            f"- Patterns: `{paired['patterns']}`",
            f"- Non-empty files: `{paired['nonempty_files']}`",
            "",
            "### optional_artifacts",
            "",
            f"- Non-empty files: `{audit['optional_artifacts']['nonempty_files']}`",
            "",
            "## Validation",
            "",
        ]
    )
    validation = audit["validation"]
    if validation["problems"]:
        lines.append("Problems:")
        lines.extend(f"- {problem}" for problem in validation["problems"])
        lines.append("")
    if validation["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in validation["warnings"])
        lines.append("")
    if not validation["problems"] and not validation["warnings"]:
        lines.append("No validation problems or warnings were reported.")
        lines.append("")

    lines.extend(["## Notes", ""])
    if audit["notes"]:
        lines.extend(f"- {note}" for note in audit["notes"])
    else:
        lines.append("- No additional audit notes.")
    lines.append("")
    return "\n".join(lines)


def write_artifact_audit(
    result_dir: Path,
    *,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
    config_path: Path = ROOT / "config" / "trials.yaml",
    allow_dry_run: bool = False,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    audit = audit_result_dir(
        result_dir,
        config_path=config_path,
        allow_dry_run=allow_dry_run,
        allow_pilot=allow_pilot,
    )
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(audit, indent=2) + "\n")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_artifact_audit(audit))
    return audit


def strict_failures(audit: dict[str, Any], required_claims: list[str]) -> list[str]:
    failures: list[str] = []
    if not audit["strict_pass"]:
        failures.append("core result-directory audit did not pass")
    for claim in required_claims:
        if not audit["readiness"][claim]:
            failures.append(f"required claim tier is not ready: {claim}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark result-directory artifacts")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "trials.yaml")
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument(
        "--require",
        action="append",
        choices=sorted(CLAIM_LABELS),
        default=[],
        help="Claim tier that must be ready when --strict is used; repeatable.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    audit = write_artifact_audit(
        args.result_dir,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        config_path=args.config,
        allow_dry_run=args.allow_dry_run,
        allow_pilot=args.allow_pilot,
    )
    if not args.json_output and not args.markdown_output:
        print(render_artifact_audit(audit))
    else:
        if args.json_output:
            print(f"Wrote {args.json_output}")
        if args.markdown_output:
            print(f"Wrote {args.markdown_output}")

    failures = strict_failures(audit, args.require)
    if args.strict and failures:
        for failure in failures:
            print(f"PROBLEM: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
