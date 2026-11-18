"""Tests for distinct, label-derivable synthetic instances."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import (  # noqa: E402
    generate_compliance_samples,
    generate_complaints,
    generate_invoices,
    generate_sql_samples,
    generate_summarization_samples,
    generate_workflow_samples,
    regenerate_gold_from_record,
)


def test_all_records_have_template_id_and_unique_sample_ids():
    generators = [
        generate_invoices(60),
        generate_complaints(40),
        generate_sql_samples(50),
        generate_summarization_samples(50),
        generate_compliance_samples(50),
        generate_workflow_samples(50),
    ]
    for records in generators:
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))
        assert all(r.get("template_id") for r in records)
        for record in records:
            assert regenerate_gold_from_record(record) == record["gold"]


def test_sql_and_compliance_are_not_tiny_template_cycles():
    sql = generate_sql_samples(50)
    compliance = generate_compliance_samples(50)
    workflow = generate_workflow_samples(50)
    assert len({r["question"] for r in sql}) == len(sql)
    assert len({r["setup_sql"] for r in sql}) == len(sql)
    assert len({r["document"] for r in compliance}) == len(compliance)
    assert len({r["text"] for r in workflow}) == len(workflow)
    # At least 8 SQL intent families and 5 workflow classes.
    assert len({r["template_id"] for r in sql}) >= 8
    assert len({r["template_id"] for r in workflow}) >= 5


def test_regenerated_jsonl_round_trip_matches_generator(tmp_path: Path):
    records = generate_sql_samples(10)
    path = tmp_path / "sql.jsonl"
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    loaded = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert loaded == records


def test_on_disk_synthetic_files_have_template_ids_when_present():
    path = ROOT / "data" / "synthetic" / "sql.jsonl"
    if not path.exists():
        return
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return
    # After regeneration every row should carry template_id.
    if "template_id" in rows[0]:
        assert all(row.get("template_id") for row in rows)
        assert len({row["question"] for row in rows}) == len(rows)
