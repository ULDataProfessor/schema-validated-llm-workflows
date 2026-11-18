"""Validate and document stress-condition manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RULES = {
    "separate_from_primary_matrix": True,
    "require_explicit_label": True,
    "do_not_rank_frameworks_from_stress": True,
}
REQUIRED_JOIN_KEY_FIELDS = {"framework", "task", "sample_id", "seed", "stress_id"}
REQUIRED_CONDITION_FIELDS = {
    "stress_id",
    "base_sample_id",
    "task",
    "family",
    "manipulation",
    "expected_invariant",
    "fault",
}
EXPECTED_FAMILIES = {
    "repeated_execution",
    "input_perturbation",
    "tool_or_api_fault",
    "schema_drift",
}
TASK_DATA_FILES = {
    "extraction": ("extraction.jsonl", "complaints.jsonl", "invoices.jsonl"),
    "sql_generation": ("sql.jsonl",),
    "summarization": ("summarization.jsonl",),
    "compliance_qa": ("compliance_qa.jsonl",),
    "multi_step": ("multi_step.jsonl",),
}


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return data


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _condition_list(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(manifest.get("stress_conditions"), list):
        return [item for item in manifest["stress_conditions"] if isinstance(item, dict)]
    if isinstance(manifest.get("conditions"), list):
        return [item for item in manifest["conditions"] if isinstance(item, dict)]
    return []


def load_sample_ids(data_dir: Path = ROOT / "data" / "synthetic") -> dict[str, set[str]]:
    sample_ids: dict[str, set[str]] = {task: set() for task in TASK_DATA_FILES}
    for task, filenames in TASK_DATA_FILES.items():
        for filename in filenames:
            path = data_dir / filename
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = row.get("id")
                if isinstance(sample_id, str):
                    sample_ids[task].add(sample_id)
    return sample_ids


def validate_manifest(
    manifest: dict[str, Any],
    trials: dict[str, Any],
    *,
    sample_ids: dict[str, set[str]] | None = None,
    validate_sample_ids: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    configured_tasks = set(_string_list(trials.get("tasks")))
    if not configured_tasks:
        errors.append("trials configuration has no task list")

    rules = manifest.get("rules")
    if not isinstance(rules, dict):
        errors.append("manifest is missing a rules mapping")
        rules = {}
    for rule, expected in REQUIRED_RULES.items():
        if rules.get(rule) is not expected:
            errors.append(f"rules.{rule} must be {expected!r}")

    join_keys = set(_string_list(rules.get("join_key_fields")))
    missing_join_keys = sorted(REQUIRED_JOIN_KEY_FIELDS - join_keys)
    if missing_join_keys:
        errors.append(f"rules.join_key_fields missing {missing_join_keys}")

    families = manifest.get("families")
    if not isinstance(families, dict):
        errors.append("manifest is missing a families mapping")
        families = {}
    missing_families = sorted(EXPECTED_FAMILIES - set(families))
    if missing_families:
        errors.append(f"manifest families missing {missing_families}")

    conditions = _condition_list(manifest)
    if not conditions:
        errors.append("manifest has no stress_conditions")

    if sample_ids is None and validate_sample_ids:
        sample_ids = load_sample_ids()
    sample_ids = sample_ids or {}

    stress_ids: list[str] = []
    summaries: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        missing_fields = sorted(REQUIRED_CONDITION_FIELDS - set(condition))
        stress_id = str(condition.get("stress_id", f"<missing:{index}>"))
        stress_ids.append(stress_id)
        if missing_fields:
            errors.append(f"{stress_id} missing required fields {missing_fields}")

        task = condition.get("task")
        if task not in configured_tasks:
            errors.append(f"{stress_id} uses unknown task {task!r}")

        family = condition.get("family")
        if family not in EXPECTED_FAMILIES:
            errors.append(f"{stress_id} uses unknown stress family {family!r}")

        base_sample_id = condition.get("base_sample_id")
        if not isinstance(base_sample_id, str) or not base_sample_id:
            errors.append(f"{stress_id} has no base_sample_id")
        elif (
            validate_sample_ids
            and isinstance(task, str)
            and task in sample_ids
            and sample_ids[task]
            and base_sample_id not in sample_ids[task]
        ):
            errors.append(
                f"{stress_id} base_sample_id {base_sample_id!r} not found for task {task!r}"
            )

        if not condition.get("manipulation"):
            errors.append(f"{stress_id} has an empty manipulation")
        if not isinstance(condition.get("expected_invariant"), dict):
            errors.append(f"{stress_id} expected_invariant must be a mapping")

        fault = condition.get("fault")
        if not isinstance(fault, dict):
            errors.append(f"{stress_id} fault must be a mapping")
            fault = {}
        fault_type = fault.get("type")
        if family == "tool_or_api_fault" and not fault_type:
            errors.append(f"{stress_id} tool_or_api_fault requires fault.type")
        if family != "tool_or_api_fault" and fault_type and family != "schema_drift":
            warnings.append(
                f"{stress_id} has fault.type under non-fault family {family!r}"
            )

        summaries.append(
            {
                "stress_id": stress_id,
                "task": task,
                "family": family,
                "base_sample_id": base_sample_id,
                "manipulation": condition.get("manipulation", ""),
                "fault_type": fault_type,
                "expected_invariant_keys": sorted(
                    condition.get("expected_invariant", {}).keys()
                )
                if isinstance(condition.get("expected_invariant"), dict)
                else [],
            }
        )

    duplicates = sorted({stress_id for stress_id in stress_ids if stress_ids.count(stress_id) > 1})
    if duplicates:
        errors.append(f"duplicate stress ids {duplicates}")

    family_counts = dict(Counter(item["family"] for item in summaries if item["family"]))
    missing_condition_families = sorted(EXPECTED_FAMILIES - set(family_counts))
    if missing_condition_families:
        errors.append(f"no stress condition covers families {missing_condition_families}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "configured_tasks": sorted(configured_tasks),
        "expected_families": sorted(EXPECTED_FAMILIES),
        "family_counts": family_counts,
        "condition_count": len(summaries),
        "rules": rules,
        "conditions": summaries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "ready" if report["ok"] else "not ready"
    lines = [
        "# Stress Manifest Audit",
        "",
        f"- Status: **{status}**",
        f"- Stress condition count: `{report['condition_count']}`",
        f"- Configured tasks: `{', '.join(report['configured_tasks'])}`",
        f"- Expected families: `{', '.join(report['expected_families'])}`",
        f"- Family counts: `{report['family_counts']}`",
        "",
        "## Stress Conditions",
        "",
        "| Stress ID | Family | Task | Base sample | Manipulation | Fault type | Invariant keys |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition in report["conditions"]:
        fault_type = condition["fault_type"] if condition["fault_type"] is not None else "null"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{condition['stress_id']}`",
                    f"`{condition['family']}`",
                    f"`{condition['task']}`",
                    f"`{condition['base_sample_id']}`",
                    str(condition["manipulation"]),
                    f"`{fault_type}`",
                    ", ".join(f"`{key}`" for key in condition["expected_invariant_keys"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Validation", ""])
    if report["errors"]:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in report["errors"])
        lines.append("")
    if report["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")
    if not report["errors"] and not report["warnings"]:
        lines.append("No validation problems or warnings were reported.")
    return "\n".join(lines)


def write_outputs(
    *,
    manifest: dict[str, Any],
    report: dict[str, Any],
    manifest_output: Path | None = None,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
) -> None:
    if manifest_output is not None:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(yaml.safe_dump(manifest, sort_keys=False))
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2) + "\n")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(report) + "\n")


def build_report(
    *,
    manifest_path: Path = ROOT / "config" / "stress.yaml",
    trials_path: Path = ROOT / "config" / "trials.yaml",
    validate_sample_ids: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_yaml(manifest_path)
    trials = load_yaml(trials_path)
    report = validate_manifest(
        manifest,
        trials,
        validate_sample_ids=validate_sample_ids,
    )
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a stress-condition manifest")
    parser.add_argument("--manifest", type=Path, default=ROOT / "config" / "stress.yaml")
    parser.add_argument("--trials", type=Path, default=ROOT / "config" / "trials.yaml")
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument(
        "--skip-sample-id-check",
        action="store_true",
        help="Do not validate base_sample_id values against generated synthetic data",
    )
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on validation errors")
    args = parser.parse_args()

    manifest, report = build_report(
        manifest_path=args.manifest,
        trials_path=args.trials,
        validate_sample_ids=not args.skip_sample_id_check,
    )
    write_outputs(
        manifest=manifest,
        report=report,
        manifest_output=args.manifest_output,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )

    if args.manifest_output is None and args.json_output is None and args.markdown_output is None:
        print(render_markdown(report))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
