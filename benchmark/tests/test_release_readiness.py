import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_release_readiness.py"
spec = importlib.util.spec_from_file_location("check_release_readiness", SCRIPT_PATH)
release = importlib.util.module_from_spec(spec)
sys.modules["check_release_readiness"] = release
assert spec.loader is not None
spec.loader.exec_module(release)


def test_tail_text_keeps_short_text_and_truncates_long_text() -> None:
    assert release.tail_text("short", 20) == "short"

    truncated = release.tail_text("abcdef", 3)

    assert truncated.startswith("[truncated 3 chars]")
    assert truncated.endswith("def")


def test_scan_latex_log_reports_problem_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "main.log"
    log_path.write_text("clean line\nLaTeX Error: Missing item\n")

    findings = release.scan_latex_log(log_path, root=tmp_path)

    assert findings == ["main.log:2: LaTeX Error: Missing item"]


def test_artifact_status_checks_size_threshold(tmp_path: Path) -> None:
    artifact = tmp_path / "main.pdf"
    artifact.write_bytes(b"x" * 12)

    statuses = release.artifact_status([(artifact, 10), (tmp_path / "missing.tgz", 1)])

    assert statuses[0]["ok"]
    assert statuses[0]["size_bytes"] == 12
    assert not statuses[1]["ok"]
    assert not statuses[1]["exists"]


def test_claim_record_findings_validate_expected_pilot_boundary(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "claim_records.md").write_text(
        "# Claim Records\nLive runner readiness\nDescriptive framework-task results\nblocked\n"
    )
    (result_dir / "claim_records.json").write_text(
        release.json.dumps(
            {
                "run_id": "run",
                "supported_claim_ids": ["runner_readiness"],
                "blocked_claim_ids": ["descriptive_results"],
                "records": [{"claim_id": "runner_readiness"}, {"claim_id": "descriptive_results"}],
            }
        )
        + "\n"
    )

    findings = release.claim_record_findings(
        result_dir,
        expected_supported={"runner_readiness"},
        expected_blocked={"descriptive_results"},
        root=tmp_path,
    )

    assert findings == []


def test_claim_record_findings_report_missing_and_boundary_mismatch(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "claim_records.md").write_text("# Claim Records\n")
    (result_dir / "claim_records.json").write_text(
        release.json.dumps(
            {
                "run_id": "different",
                "supported_claim_ids": ["descriptive_results"],
                "blocked_claim_ids": [],
                "records": [],
            }
        )
        + "\n"
    )

    findings = release.claim_record_findings(
        result_dir,
        expected_supported={"runner_readiness"},
        expected_blocked={"descriptive_results"},
        root=tmp_path,
    )

    assert any("run_id='different'" in finding for finding in findings)
    assert any("supported pilot claims differ" in finding for finding in findings)
    assert any("blocked pilot claims differ" in finding for finding in findings)
    assert any("missing expected phrase" in finding for finding in findings)


def test_bibliography_findings_accept_valid_cited_entries(tmp_path: Path) -> None:
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(r"\citep{smith2026valid,docs2026}" + "\n")
    bib_path.write_text(
        """
@article{smith2026valid,
  author  = {Smith, Ada},
  title   = {Valid Structured Output Evidence},
  journal = {arXiv preprint arXiv:2601.12345},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.12345}
}

@misc{docs2026,
  author       = {{Example Project}},
  title        = {Project Documentation},
  howpublished = {\\url{https://example.com/docs}},
  year         = {2026},
  note         = {Accessed 2026-06-12}
}
"""
    )

    findings = release.bibliography_findings(
        bib_path=bib_path,
        tex_paths=[tex_path],
        root=tmp_path,
    )

    assert findings == []


def test_bibliography_findings_report_missing_duplicate_and_incomplete_entries(
    tmp_path: Path,
) -> None:
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(r"\citep{missing,dup,incomplete}" + "\n")
    bib_path.write_text(
        """
@article{dup,
  title   = {First Entry},
  journal = {Journal},
  year    = {2026}
}

@misc{dup,
  title        = {Second Entry},
  howpublished = {\\url{https://example.com}},
  year         = {2026}
}

@article{incomplete,
  author = {No Title}
}
"""
    )

    findings = release.bibliography_findings(
        bib_path=bib_path,
        tex_paths=[tex_path],
        root=tmp_path,
    )

    assert any("Cited keys missing" in finding and "missing" in finding for finding in findings)
    assert any("Duplicate bibliography key: dup" in finding for finding in findings)
    assert any("incomplete is missing title" in finding for finding in findings)
    assert any("incomplete is missing year" in finding for finding in findings)
    assert any("incomplete is missing a source field" in finding for finding in findings)


def test_bibliography_findings_report_arxiv_identifier_mismatch(tmp_path: Path) -> None:
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "references.bib"
    tex_path.write_text(r"\citep{bad2026}" + "\n")
    bib_path.write_text(
        """
@article{bad2026,
  title   = {Mismatched Arxiv Metadata},
  journal = {arXiv preprint arXiv:2601.12345},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.54321}
}
"""
    )

    findings = release.bibliography_findings(
        bib_path=bib_path,
        tex_paths=[tex_path],
        root=tmp_path,
    )

    assert any("mismatched arXiv ids and DOI ids" in finding for finding in findings)


def test_claim_scope_summary_reads_generated_claim_records(tmp_path: Path) -> None:
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "claim_records.json").write_text(
        release.json.dumps(
            {
                "run_id": "run",
                "eligibility": "pilot-only",
                "claim_scope": "runner readiness only",
                "supported_claim_ids": ["runner_readiness"],
                "blocked_claim_ids": ["inferential_claims", "descriptive_results"],
            }
        )
        + "\n"
    )

    summary = release.claim_scope_summary(result_dir, root=tmp_path)

    assert summary["available"]
    assert summary["run_id"] == "run"
    assert summary["eligibility"] == "pilot-only"
    assert summary["claim_scope"] == "runner readiness only"
    assert summary["supported_claim_ids"] == ["runner_readiness"]
    assert summary["blocked_claim_ids"] == ["descriptive_results", "inferential_claims"]


def test_run_command_records_success(tmp_path: Path) -> None:
    check = release.run_command(
        "noop",
        [sys.executable, "-c", "print('ready')"],
        cwd=tmp_path,
        stdout_limit=20,
        root=tmp_path,
    )

    assert check["ok"]
    assert check["returncode"] == 0
    assert check["stdout_tail"].strip() == "ready"


def test_render_markdown_includes_failed_command_details() -> None:
    report = {
        "ok": False,
        "checks": [
            {
                "name": "example",
                "kind": "command",
                "ok": False,
                "returncode": 2,
                "cwd": "benchmark",
                "command": "python -m pytest tests",
            }
        ],
        "artifacts": [
            {
                "path": "paper/main.pdf",
                "size_bytes": 0,
                "ok": False,
            }
        ],
        "claim_scope": {
            "run_id": "pilot-run",
            "eligibility": "pilot-only",
            "claim_scope": "runner readiness only",
            "supported_claim_ids": ["runner_readiness"],
            "blocked_claim_ids": ["inferential_claims"],
        },
    }

    markdown = release.render_markdown(report)

    assert "Overall status: not ready" in markdown
    assert "returned 2" in markdown
    assert "## Claim Scope" in markdown
    assert "`runner_readiness`" in markdown
    assert "`inferential_claims`" in markdown
