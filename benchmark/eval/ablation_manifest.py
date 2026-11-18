"""Validate and document the pre-specified ablation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RULES = {
    "separate_from_primary_matrix": True,
    "require_explicit_label": True,
    "do_not_rank_frameworks_from_ablations": True,
}
REQUIRED_TRIAL_KEY_FIELDS = {
    "framework",
    "task",
    "sample_id",
    "seed",
    "ablation_id",
    "condition",
}
REQUIRED_ABLATION_FIELDS = {
    "id",
    "purpose",
    "primary_question",
    "comparison",
    "tasks",
    "primary_metrics",
    "claim_boundary",
}
REQUIRED_COMPARISON_FIELDS = {"condition", "description"}
EXPECTED_ABLATION_IDS = {
    "provider_native_vs_application_validation",
    "zero_retry_vs_final_validity",
    "domain_validators_enabled",
    "typed_dependencies_and_tools",
    "secondary_model_subset",
}
KNOWN_METRICS = {
    "schema_valid",
    "zero_retry_success",
    "quality_score",
    "retries",
    "latency_ms",
    "total_tokens",
    "error_type_counts",
    "tool_selection_accuracy",
    "citation_support",
    "trace_completeness",
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


def validate_manifest(
    manifest: dict[str, Any],
    trials: dict[str, Any],
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

    trial_key_fields = set(_string_list(rules.get("trial_key_fields")))
    missing_trial_keys = sorted(REQUIRED_TRIAL_KEY_FIELDS - trial_key_fields)
    if missing_trial_keys:
        errors.append(f"rules.trial_key_fields missing {missing_trial_keys}")

    ablations = manifest.get("ablations")
    if not isinstance(ablations, list):
        errors.append("manifest is missing an ablations list")
        ablations = []

    ids: list[str] = []
    summaries: list[dict[str, Any]] = []
    for index, item in enumerate(ablations):
        if not isinstance(item, dict):
            errors.append(f"ablations[{index}] is not a mapping")
            continue

        missing_fields = sorted(REQUIRED_ABLATION_FIELDS - set(item))
        ablation_id = str(item.get("id", f"<missing:{index}>"))
        ids.append(ablation_id)
        if missing_fields:
            errors.append(f"{ablation_id} missing required fields {missing_fields}")

        tasks = _string_list(item.get("tasks"))
        unknown_tasks = sorted(set(tasks) - configured_tasks)
        if unknown_tasks:
            errors.append(f"{ablation_id} uses unknown tasks {unknown_tasks}")
        if not tasks:
            errors.append(f"{ablation_id} has no tasks")

        metrics = _string_list(item.get("primary_metrics"))
        unknown_metrics = sorted(set(metrics) - KNOWN_METRICS)
        if unknown_metrics:
            errors.append(f"{ablation_id} uses unknown primary metrics {unknown_metrics}")
        if not metrics:
            errors.append(f"{ablation_id} has no primary metrics")

        comparison = item.get("comparison")
        if not isinstance(comparison, list) or len(comparison) < 2:
            errors.append(f"{ablation_id} must define at least two comparison conditions")
            comparison = []

        conditions: list[str] = []
        for condition_index, condition in enumerate(comparison):
            if not isinstance(condition, dict):
                errors.append(
                    f"{ablation_id}.comparison[{condition_index}] is not a mapping"
                )
                continue
            missing_condition_fields = sorted(REQUIRED_COMPARISON_FIELDS - set(condition))
            if missing_condition_fields:
                errors.append(
                    f"{ablation_id}.comparison[{condition_index}] missing "
                    f"{missing_condition_fields}"
                )
            name = condition.get("condition")
            if isinstance(name, str) and name.strip():
                conditions.append(name)
            else:
                errors.append(f"{ablation_id}.comparison[{condition_index}] has no condition")

        if len(conditions) != len(set(conditions)):
            errors.append(f"{ablation_id} has duplicate condition names")

        if not item.get("purpose"):
            errors.append(f"{ablation_id} has an empty purpose")
        if not item.get("primary_question"):
            errors.append(f"{ablation_id} has an empty primary_question")
        if not item.get("claim_boundary"):
            errors.append(f"{ablation_id} has an empty claim_boundary")

        summaries.append(
            {
                "id": ablation_id,
                "purpose": item.get("purpose", ""),
                "primary_question": item.get("primary_question", ""),
                "tasks": tasks,
                "conditions": conditions,
                "primary_metrics": metrics,
                "claim_boundary": item.get("claim_boundary", ""),
            }
        )

    duplicate_ids = sorted({ablation_id for ablation_id in ids if ids.count(ablation_id) > 1})
    if duplicate_ids:
        errors.append(f"duplicate ablation ids {duplicate_ids}")

    missing_expected = sorted(EXPECTED_ABLATION_IDS - set(ids))
    if missing_expected:
        errors.append(f"missing expected ablations {missing_expected}")

    extra_ids = sorted(set(ids) - EXPECTED_ABLATION_IDS)
    if extra_ids:
        warnings.append(f"extra ablations are present and should be documented: {extra_ids}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "configured_tasks": sorted(configured_tasks),
        "known_metrics": sorted(KNOWN_METRICS),
        "required_ablation_ids": sorted(EXPECTED_ABLATION_IDS),
        "configured_ablation_ids": ids,
        "ablation_count": len(summaries),
        "rules": rules,
        "ablations": summaries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "ready" if report["ok"] else "not ready"
    lines = [
        "# Ablation Manifest Audit",
        "",
        f"- Status: **{status}**",
        f"- Ablation count: `{report['ablation_count']}`",
        f"- Configured tasks: `{', '.join(report['configured_tasks'])}`",
        f"- Required ablations: `{', '.join(report['required_ablation_ids'])}`",
        "",
        "## Mechanism Checks",
        "",
        "| Ablation | Conditions | Tasks | Primary metrics | Claim boundary |",
        "|---|---|---|---|---|",
    ]
    for item in report["ablations"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['id']}`",
                    ", ".join(f"`{condition}`" for condition in item["conditions"]),
                    ", ".join(f"`{task}`" for task in item["tasks"]),
                    ", ".join(f"`{metric}`" for metric in item["primary_metrics"]),
                    str(item["claim_boundary"]),
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


def write_report(
    report: dict[str, Any],
    *,
    json_output: Path | None = None,
    markdown_output: Path | None = None,
) -> None:
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2) + "\n")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(report) + "\n")


def build_report(
    *,
    manifest_path: Path = ROOT / "config" / "ablations.yaml",
    trials_path: Path = ROOT / "config" / "trials.yaml",
) -> dict[str, Any]:
    return validate_manifest(load_yaml(manifest_path), load_yaml(trials_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the ablation manifest")
    parser.add_argument("--manifest", type=Path, default=ROOT / "config" / "ablations.yaml")
    parser.add_argument("--trials", type=Path, default=ROOT / "config" / "trials.yaml")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on validation errors")
    args = parser.parse_args()

    report = build_report(manifest_path=args.manifest, trials_path=args.trials)
    write_report(
        report,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )

    if args.json_output is None and args.markdown_output is None:
        print(render_markdown(report))
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
