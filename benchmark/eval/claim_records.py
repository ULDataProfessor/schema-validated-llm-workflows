"""Generate typed claim records from artifact-audit readiness evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from eval.artifact_audit import (
    ARTIFACT_GROUPS,
    CLAIM_LABELS,
    ROOT,
    audit_result_dir,
    ready_text,
)


ClaimStatus = Literal["supported", "blocked"]


class ClaimRecord(BaseModel):
    claim_id: str
    claim_label: str
    status: ClaimStatus
    evidence_tier: str
    supported_claim: str
    readiness_note: str
    required_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    blocked_claims: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    reviewer_note: str | None = None


CLAIM_TEXT = {
    "software_path": "The dry-run software path is executable and produces core artifacts.",
    "runner_readiness": (
        "The provider-backed framework-task request paths are executable under the "
        "tested environment."
    ),
    "descriptive_results": (
        "The run can support descriptive framework-task comparisons for the configured matrix."
    ),
    "inferential_claims": (
        "The run can support paired inferential framework comparisons on shared samples."
    ),
    "failure_analysis": (
        "The run can support deterministic qualitative failure-mode inspection."
    ),
    "cost_analysis": "The run can support token-normalized and price-normalized cost analysis.",
    "mechanism_ablation_manifest": (
        "The repository can support pre-specified mechanism-ablation design claims."
    ),
    "stress_manifest": (
        "The repository can support pre-specified stress-extension design claims."
    ),
    "human_audit_packet": "The run can support human-audit packet generation claims.",
    "semantic_human_audit": (
        "The run can support completed human-review semantic-audit claims."
    ),
    "maintainability_review": (
        "The run can support completed maintainability/schema-change review claims."
    ),
    "environment_metadata": (
        "The run can support environment, package-version, and config-hash provenance claims."
    ),
    "run_provenance": (
        "The run can support command-line and timestamp provenance claims."
    ),
}

CLAIM_ARTIFACT_GROUPS = {
    "software_path": ("core",),
    "runner_readiness": ("core",),
    "descriptive_results": ("core",),
    "failure_analysis": ("failure_analysis",),
    "cost_analysis": ("cost_analysis",),
    "mechanism_ablation_manifest": ("ablation_manifest",),
    "stress_manifest": ("stress_manifest",),
    "human_audit_packet": ("human_audit_packet",),
    "semantic_human_audit": ("human_audit_packet", "human_audit_summary"),
    "maintainability_review": ("maintainability",),
}

CLAIM_EXTRA_ARTIFACTS = {
    "inferential_claims": ("paired_*.json",),
    "environment_metadata": (
        "meta.json:environment.python_version",
        "meta.json:environment.platform",
        "meta.json:environment.package_versions",
        "meta.json:environment.config_hashes",
    ),
    "run_provenance": (
        "meta.json:provenance.started_at_utc",
        "meta.json:provenance.working_directory",
        "meta.json:provenance.command_argv",
        "meta.json:provenance.args",
    ),
}


def evidence_tier(claim_id: str, audit: dict[str, Any]) -> str:
    if claim_id == "software_path":
        return "dry_run"
    if claim_id == "runner_readiness":
        return "pilot" if audit["run_flags"]["pilot"] else "full_matrix"
    if claim_id in {"descriptive_results", "inferential_claims"}:
        return "full_matrix"
    if claim_id == "mechanism_ablation_manifest":
        return "ablation"
    if claim_id == "stress_manifest":
        return "stress"
    if claim_id in {"human_audit_packet", "semantic_human_audit"}:
        return "human_audit"
    if claim_id == "maintainability_review":
        return "maintainability"
    if claim_id in {"environment_metadata", "run_provenance"}:
        return "metadata"
    return claim_id


def required_artifacts_for(claim_id: str) -> list[str]:
    artifacts: list[str] = []
    for group_name in CLAIM_ARTIFACT_GROUPS.get(claim_id, ()):
        artifacts.extend(ARTIFACT_GROUPS[group_name])
    artifacts.extend(CLAIM_EXTRA_ARTIFACTS.get(claim_id, ()))
    return artifacts


def missing_artifacts_for(claim_id: str, audit: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for group_name in CLAIM_ARTIFACT_GROUPS.get(claim_id, ()):
        missing.extend(audit["missing_artifacts"].get(group_name, []))
    if claim_id == "inferential_claims" and audit["paired_analysis"]["count"] == 0:
        missing.extend(CLAIM_EXTRA_ARTIFACTS[claim_id])
    if claim_id in {"environment_metadata", "run_provenance"} and not audit["readiness"][claim_id]:
        missing.extend(CLAIM_EXTRA_ARTIFACTS[claim_id])
    return missing


def build_claim_records(audit: dict[str, Any]) -> list[ClaimRecord]:
    blocked_claims = [
        CLAIM_LABELS[claim_id]
        for claim_id, ready in audit["readiness"].items()
        if not ready
    ]
    run_id = str(audit["run_id"])
    records: list[ClaimRecord] = []
    for claim_id, label in CLAIM_LABELS.items():
        ready = bool(audit["readiness"][claim_id])
        records.append(
            ClaimRecord(
                claim_id=claim_id,
                claim_label=label,
                status="supported" if ready else "blocked",
                evidence_tier=evidence_tier(claim_id, audit),
                supported_claim=CLAIM_TEXT[claim_id],
                readiness_note=audit["readiness_notes"][claim_id],
                required_artifacts=required_artifacts_for(claim_id),
                missing_artifacts=missing_artifacts_for(claim_id, audit),
                blocked_claims=[] if not ready else blocked_claims,
                run_ids=[run_id],
                reviewer_note=None if ready else audit["readiness_notes"][claim_id],
            )
        )
    return records


def build_report(audit: dict[str, Any]) -> dict[str, Any]:
    records = build_claim_records(audit)
    return {
        "run_id": audit["run_id"],
        "result_dir": audit["result_dir"],
        "eligibility": audit["eligibility"],
        "claim_scope": audit["claim_scope"],
        "records": [record.model_dump() for record in records],
        "supported_claim_ids": [
            record.claim_id for record in records if record.status == "supported"
        ],
        "blocked_claim_ids": [
            record.claim_id for record in records if record.status == "blocked"
        ],
    }


def render_claim_records(report: dict[str, Any]) -> str:
    lines = [
        f"# Claim Records: {report['run_id']}",
        "",
        "## Scope",
        "",
        f"- Result directory: `{report['result_dir']}`",
        f"- Eligibility: **{report['eligibility']}**",
        f"- Claim scope: {report['claim_scope']}",
        "",
        "## Records",
        "",
        "| Claim | Status | Evidence tier | Evidence note | Missing artifacts |",
        "|---|---|---|---|---|",
    ]
    for record in report["records"]:
        missing = ", ".join(f"`{item}`" for item in record["missing_artifacts"]) or "none"
        lines.append(
            f"| {record['claim_label']} | {record['status']} | "
            f"`{record['evidence_tier']}` | {record['readiness_note']} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Supported Claims",
            "",
        ]
    )
    supported = [record for record in report["records"] if record["status"] == "supported"]
    if supported:
        for record in supported:
            lines.append(f"- `{record['claim_id']}`: {record['supported_claim']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Blocked Claims", ""])
    blocked = [record for record in report["records"] if record["status"] == "blocked"]
    if blocked:
        for record in blocked:
            lines.append(f"- `{record['claim_id']}`: {record['reviewer_note']}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def write_claim_records(
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
    report = build_report(audit)
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report, indent=2) + "\n")
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_claim_records(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate typed claim records from an artifact audit"
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "trials.yaml")
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args()

    report = write_claim_records(
        args.result_dir,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        config_path=args.config,
        allow_dry_run=args.allow_dry_run,
        allow_pilot=args.allow_pilot,
    )
    if not args.json_output and not args.markdown_output:
        print(render_claim_records(report))
    else:
        if args.json_output:
            print(f"Wrote {args.json_output}")
        if args.markdown_output:
            print(f"Wrote {args.markdown_output}")

    if not any(record["status"] == "supported" for record in report["records"]):
        print("PROBLEM: no supported claims were found")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
