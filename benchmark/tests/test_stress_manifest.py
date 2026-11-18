import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.stress_manifest import (  # noqa: E402
    EXPECTED_FAMILIES,
    build_report,
    render_markdown,
    validate_manifest,
    write_outputs,
)


def test_stress_manifest_covers_expected_families_and_samples() -> None:
    _, report = build_report()

    assert report["ok"]
    assert not report["errors"]
    assert set(report["family_counts"]) == EXPECTED_FAMILIES
    assert report["rules"]["separate_from_primary_matrix"] is True
    assert report["rules"]["require_explicit_label"] is True
    assert report["rules"]["do_not_rank_frameworks_from_stress"] is True


def test_stress_manifest_rejects_unknown_task_family_and_sample() -> None:
    trials = yaml.safe_load((ROOT / "config" / "trials.yaml").read_text())
    manifest = yaml.safe_load((ROOT / "config" / "stress.yaml").read_text())
    manifest["stress_conditions"][0]["task"] = "unknown_task"
    manifest["stress_conditions"][1]["family"] = "unknown_family"
    manifest["stress_conditions"][2]["base_sample_id"] = "missing_sample"

    report = validate_manifest(manifest, trials)

    assert not report["ok"]
    assert any("unknown_task" in error for error in report["errors"])
    assert any("unknown_family" in error for error in report["errors"])
    assert any("missing_sample" in error for error in report["errors"])


def test_stress_manifest_requires_fault_type_for_tool_fault() -> None:
    trials = yaml.safe_load((ROOT / "config" / "trials.yaml").read_text())
    manifest = yaml.safe_load((ROOT / "config" / "stress.yaml").read_text())
    for condition in manifest["stress_conditions"]:
        if condition["family"] == "tool_or_api_fault":
            condition["fault"]["type"] = None
            break

    report = validate_manifest(manifest, trials)

    assert not report["ok"]
    assert any("requires fault.type" in error for error in report["errors"])


def test_stress_manifest_writes_manifest_json_and_markdown(tmp_path: Path) -> None:
    manifest, report = build_report()
    manifest_output = tmp_path / "stress_manifest.yaml"
    json_output = tmp_path / "stress_manifest_audit.json"
    markdown_output = tmp_path / "stress_manifest_audit.md"

    write_outputs(
        manifest=manifest,
        report=report,
        manifest_output=manifest_output,
        json_output=json_output,
        markdown_output=markdown_output,
    )

    assert yaml.safe_load(manifest_output.read_text())["version"] == 1
    assert json.loads(json_output.read_text())["ok"] is True
    markdown = markdown_output.read_text()
    assert markdown.startswith("# Stress Manifest Audit")
    assert "sql_timeout_001" in markdown
    assert "No validation problems" in render_markdown(report)
