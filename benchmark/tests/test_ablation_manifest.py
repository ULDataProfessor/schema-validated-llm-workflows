import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.ablation_manifest import (  # noqa: E402
    EXPECTED_ABLATION_IDS,
    build_report,
    render_markdown,
    validate_manifest,
    write_report,
)


def test_ablation_manifest_has_unique_ids_and_known_tasks() -> None:
    report = build_report()

    assert report["ok"]
    assert not report["errors"]
    assert set(report["configured_ablation_ids"]) == EXPECTED_ABLATION_IDS
    assert report["rules"]["separate_from_primary_matrix"] is True
    assert report["rules"]["require_explicit_label"] is True
    assert report["rules"]["do_not_rank_frameworks_from_ablations"] is True


def test_ablation_manifest_rejects_missing_required_ablation() -> None:
    trials = yaml.safe_load((ROOT / "config" / "trials.yaml").read_text())
    manifest = yaml.safe_load((ROOT / "config" / "ablations.yaml").read_text())
    manifest["ablations"] = [
        item
        for item in manifest["ablations"]
        if item["id"] != "typed_dependencies_and_tools"
    ]

    report = validate_manifest(manifest, trials)

    assert not report["ok"]
    assert any("typed_dependencies_and_tools" in error for error in report["errors"])


def test_ablation_manifest_rejects_unknown_metric_and_task() -> None:
    trials = yaml.safe_load((ROOT / "config" / "trials.yaml").read_text())
    manifest = yaml.safe_load((ROOT / "config" / "ablations.yaml").read_text())
    manifest["ablations"][0]["tasks"].append("unknown_task")
    manifest["ablations"][0]["primary_metrics"].append("unknown_metric")

    report = validate_manifest(manifest, trials)

    assert not report["ok"]
    assert any("unknown_task" in error for error in report["errors"])
    assert any("unknown_metric" in error for error in report["errors"])


def test_ablation_manifest_writes_json_and_markdown(tmp_path: Path) -> None:
    report = build_report()
    json_output = tmp_path / "ablation_manifest_audit.json"
    markdown_output = tmp_path / "ablation_manifest_audit.md"

    write_report(report, json_output=json_output, markdown_output=markdown_output)

    assert json.loads(json_output.read_text())["ok"] is True
    markdown = markdown_output.read_text()
    assert markdown.startswith("# Ablation Manifest Audit")
    assert "provider_native_vs_application_validation" in markdown
    assert "No validation problems" in render_markdown(report)
