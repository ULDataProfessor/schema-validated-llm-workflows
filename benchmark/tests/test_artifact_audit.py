import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.artifact_audit import audit_result_dir, render_artifact_audit, write_artifact_audit  # noqa: E402


def _write_config(root: Path) -> Path:
    config_path = root / "trials.yaml"
    config_path.write_text(
        """
seeds: [42, 43]
sample_limits:
  extraction: 1
frameworks:
  - pydantic_ai
  - plain_sdk
tasks:
  - extraction
"""
    )
    return config_path


def _write_result_dir(
    root: Path,
    *,
    pilot: bool = False,
    dry_run: bool = False,
    n: int = 2,
) -> Path:
    result_dir = root / "results" / "run"
    result_dir.mkdir(parents=True)
    frameworks = ["dry_run"] if dry_run else ["pydantic_ai", "plain_sdk"]
    seeds = [42] if pilot else [42, 43]
    meta = {
        "run_id": "run",
        "model": "openai:gpt-4o-mini",
        "seed": 42 if pilot else None,
        "seeds": seeds,
        "dry_run": dry_run,
        "pilot": pilot,
        "frameworks": frameworks,
        "tasks": ["extraction"],
        "task_limits": {"extraction": 1},
        "planned_trials": len(frameworks) * n,
        "n_trials": len(frameworks) * n,
        "environment": {
            "python_version": "3.13.0",
            "platform": "test-platform",
            "package_versions": {"pydantic": "2.10.0"},
            "config_hashes": {"config/trials.yaml": "abc123"},
        },
        "provenance": {
            "started_at_utc": "2026-06-12T00:00:00Z",
            "completed_at_utc": "2026-06-12T00:01:00Z",
            "working_directory": str(root),
            "command_argv": ["python", "scripts/run_benchmark.py", "--all"],
            "args": {"all": True, "run_id": "run"},
        },
    }
    summary = {
        f"{framework}::extraction": {
            "n": n,
            "schema_pass_rate": 1.0,
            "zero_retry_success_rate": 0.5,
            "mean_quality_score": 0.75,
            "quality_metric": "field_macro_accuracy",
            "mean_retries": 1.0,
            "p95_latency_ms": 1200.0,
            "error_count": 0,
        }
        for framework in frameworks
    }
    (result_dir / "meta.json").write_text(json.dumps(meta))
    (result_dir / "summary.json").write_text(json.dumps(summary))
    (result_dir / "trials.jsonl").write_text("{}\n")
    (result_dir / "tables.tex").write_text("% generated\n")
    (result_dir / "result_card.md").write_text("# result card\n")
    return result_dir


def _write_secondary_artifacts(result_dir: Path) -> None:
    for filename in (
        "failure_examples.md",
        "failure_examples.json",
        "judge_triage.md",
        "judge_triage.json",
        "human_audit.md",
        "human_audit.csv",
        "human_audit.json",
        "human_audit_summary.md",
        "cost_analysis.md",
        "cost_analysis.json",
        "paired_pydantic_ai_plain_sdk.json",
        "ablation_manifest_audit.md",
        "ablation_manifest_audit.json",
        "stress_manifest.yaml",
        "stress_manifest_audit.md",
        "stress_manifest_audit.json",
        "maintainability.md",
        "maintainability.csv",
        "maintainability.json",
        "maintainability_summary.md",
        "maintainability_summary.json",
    ):
        (result_dir / filename).write_text("{}\n")
    (result_dir / "human_audit_summary.json").write_text(
        json.dumps({"n_completed_rows": 3}) + "\n"
    )
    (result_dir / "maintainability_summary.json").write_text(
        json.dumps({"frameworks": {"pydantic_ai": {"n_completed": 2}}}) + "\n"
    )
    (result_dir / "stress_manifest_audit.json").write_text(json.dumps({"ok": True}) + "\n")


def test_artifact_audit_marks_core_full_run_without_secondary_claims(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path)

    audit = audit_result_dir(result_dir, config_path=config_path)

    assert audit["strict_pass"]
    assert audit["readiness"]["descriptive_results"]
    assert audit["readiness"]["environment_metadata"]
    assert audit["readiness"]["run_provenance"]
    assert not audit["readiness"]["inferential_claims"]
    assert not audit["readiness"]["failure_analysis"]
    assert not audit["readiness"]["cost_analysis"]
    assert not audit["readiness"]["mechanism_ablation_manifest"]
    assert not audit["readiness"]["stress_manifest"]
    assert not audit["readiness"]["human_audit_packet"]
    assert not audit["readiness"]["maintainability_review"]
    assert audit["missing_artifacts"]["failure_analysis"] == [
        "failure_examples.md",
        "failure_examples.json",
    ]


def test_artifact_audit_marks_secondary_claims_ready_when_files_exist(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path)
    _write_secondary_artifacts(result_dir)

    audit = audit_result_dir(result_dir, config_path=config_path)

    assert audit["readiness"]["descriptive_results"]
    assert audit["readiness"]["inferential_claims"]
    assert audit["readiness"]["failure_analysis"]
    assert audit["readiness"]["cost_analysis"]
    assert audit["readiness"]["mechanism_ablation_manifest"]
    assert audit["readiness"]["stress_manifest"]
    assert audit["stress_manifest"]["audit_ok"]
    assert audit["readiness"]["human_audit_packet"]
    assert audit["readiness"]["semantic_human_audit"]
    assert audit["readiness"]["maintainability_review"]
    assert audit["maintainability"]["completed_rows"] == 2
    assert audit["paired_analysis"]["nonempty_files"] == ["paired_pydantic_ai_plain_sdk.json"]
    assert audit["optional_artifacts"]["nonempty_files"] == [
        "judge_triage.json",
        "judge_triage.md",
    ]


def test_artifact_audit_allows_pilot_but_blocks_framework_ranking_claims(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)

    audit = audit_result_dir(result_dir, config_path=config_path, allow_pilot=True)

    assert audit["validation"]["eligible"]
    assert audit["eligibility"] == "pilot-only"
    assert audit["readiness"]["runner_readiness"]
    assert audit["readiness"]["run_provenance"]
    assert not audit["readiness"]["descriptive_results"]
    assert not audit["readiness"]["inferential_claims"]
    assert "Pilot rows support runner-readiness checks" in " ".join(audit["notes"])


def test_write_artifact_audit_creates_json_and_markdown(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path)
    json_output = result_dir / "artifact_audit.json"
    markdown_output = result_dir / "artifact_audit.md"

    audit = write_artifact_audit(
        result_dir,
        json_output=json_output,
        markdown_output=markdown_output,
        config_path=config_path,
    )

    assert audit["strict_pass"]
    assert json.loads(json_output.read_text())["run_id"] == "run"
    markdown = markdown_output.read_text()
    assert markdown.startswith("# Artifact Audit: run")
    assert "Descriptive framework-task results" in markdown


def test_render_artifact_audit_includes_validation_problems(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)

    audit = audit_result_dir(result_dir, config_path=config_path)
    markdown = render_artifact_audit(audit)

    assert not audit["strict_pass"]
    assert "Run is pilot=true; pilot runs are not eligible for final empirical tables" in markdown
