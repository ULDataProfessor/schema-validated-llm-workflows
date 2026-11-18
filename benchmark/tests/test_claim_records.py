import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.claim_records import build_report, write_claim_records  # noqa: E402
from tests.test_artifact_audit import (  # noqa: E402
    _write_config,
    _write_result_dir,
    _write_secondary_artifacts,
)
from eval.artifact_audit import audit_result_dir  # noqa: E402


def test_claim_records_allow_pilot_readiness_but_block_rankings(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)
    audit = audit_result_dir(result_dir, config_path=config_path, allow_pilot=True)

    report = build_report(audit)
    records = {record["claim_id"]: record for record in report["records"]}

    assert records["runner_readiness"]["status"] == "supported"
    assert records["runner_readiness"]["evidence_tier"] == "pilot"
    assert records["descriptive_results"]["status"] == "blocked"
    assert records["inferential_claims"]["status"] == "blocked"
    assert "Descriptive framework-task results" in records["runner_readiness"]["blocked_claims"]


def test_claim_records_full_run_with_secondary_artifacts_supports_audit_claims(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path)
    _write_secondary_artifacts(result_dir)
    audit = audit_result_dir(result_dir, config_path=config_path)

    report = build_report(audit)
    records = {record["claim_id"]: record for record in report["records"]}

    assert records["descriptive_results"]["status"] == "supported"
    assert records["inferential_claims"]["status"] == "supported"
    assert records["failure_analysis"]["status"] == "supported"
    assert records["semantic_human_audit"]["status"] == "supported"
    assert records["maintainability_review"]["status"] == "supported"


def test_write_claim_records_creates_json_and_markdown(tmp_path):
    config_path = _write_config(tmp_path)
    result_dir = _write_result_dir(tmp_path, pilot=True, n=1)
    json_output = result_dir / "claim_records.json"
    markdown_output = result_dir / "claim_records.md"

    report = write_claim_records(
        result_dir,
        json_output=json_output,
        markdown_output=markdown_output,
        config_path=config_path,
        allow_pilot=True,
    )

    assert report["run_id"] == "run"
    assert json.loads(json_output.read_text())["supported_claim_ids"] == [
        "runner_readiness",
        "environment_metadata",
        "run_provenance",
    ]
    markdown = markdown_output.read_text()
    assert markdown.startswith("# Claim Records: run")
    assert "runner-readiness" in markdown or "runner-readiness" in markdown.lower()
