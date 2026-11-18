# Human Audit Protocol

This protocol defines how the benchmark creates reviewer packets for semantic checks
that are not fully captured by deterministic metrics.

## Purpose

Automated metrics are the primary scoring layer whenever the task has explicit
ground truth, executable SQL checks, citation-span checks, or source-fact keys.
Human review is used as a calibration layer for cases where a valid structured
object may still be misleading, incomplete, weakly supported, or procedurally
unsafe.

Human-audit results should not replace the pre-specified primary metrics. They
support interpretation of semantic error modes, hallucination-like failures,
borderline citations, ambiguous business requests, and review-routing decisions.

LLM-as-a-judge labels, if collected, are treated as triage signals rather than
ground truth. They may help prioritize ambiguous cases for review, but they must
not override deterministic evidence checks, state-grounded workflow checks, or
completed human-review summaries.

Optional judge-assisted triage can be generated from archived rows without any
live model calls:

```bash
python -m eval.judge_triage results/<run_id>/trials.jsonl \
  --json-output results/<run_id>/judge_triage.json \
  --markdown-output results/<run_id>/judge_triage.md
```

## Packet Generation

Generate the packet from archived trial rows:

```bash
cd benchmark
python -m eval.human_audit results/<run_id>/trials.jsonl \
  --json-output results/<run_id>/human_audit.json \
  --csv-output results/<run_id>/human_audit.csv \
  --markdown-output results/<run_id>/human_audit.md
```

The script selects deterministic diagnostic cases per task, plus passing control
cases. Diagnostic cases prioritize schema-invalid rows, low task-aware quality
scores, citation-support issues, SQL execution/result failures, summarization
fact misses, workflow decision errors, retries, and runtime errors. Passing
controls keep reviewers calibrated and help estimate whether the audit process is
overly sensitive to visible model imperfections.

Raw model output is omitted by default. Use `--include-raw` only for synthetic,
redacted, or otherwise approved traces.

After reviewers complete the CSV or JSON fields, summarize the completed review
file:

```bash
python -m eval.human_audit results/<run_id>/human_audit.csv \
  --summarize \
  --summary-output results/<run_id>/human_audit_summary.json \
  --summary-markdown-output results/<run_id>/human_audit_summary.md
```

The summary reports packet completion, task and case-type counts, reviewer
counts, human-correct and human-supported rates, adjudication flags, severity
counts, raw pairwise agreement, chance agreement, and Cohen's kappa for cases
with multiple reviewer rows. Kappa is a calibration diagnostic, not an automatic
acceptance threshold; when reviewer labels have no marginal variation, raw
agreement and disagreement lists should be interpreted directly.

## Reviewer Procedure

Reviewers should retrieve the source input and ground truth using the packet's
`task` and `sample_id` fields. Reviewers then complete:

- `human_correct`: whether the final answer/output is acceptable for the task.
- `human_supported`: whether claims are supported by source data or tool state.
- `human_error_category`: short failure label, if incorrect.
- `human_severity`: low, medium, high, or critical business impact.
- `needs_adjudication`: whether a second reviewer or domain expert is required.
- `reviewer_id`: pseudonymous reviewer identifier.
- `reviewer_notes`: concise rationale.

For compliance QA, reviewers should inspect whether citations are sufficient, not
only whether a cited quote appears in the policy document. For summarization,
reviewers should check material omissions and numeric consistency. For SQL,
reviewers should check intent alignment in addition to execution success. For
multi-step workflows, reviewers should check whether the stated final disposition
matches the tool trace and whether review routing was appropriate.

## Reporting

The paper should report the number of reviewed cases, the selection rule, whether
reviewers were blinded to framework labels, raw agreement and kappa when multiple
reviewers are used, adjudication rules, and representative examples.

Human-audit claims require archived packets, completed review files, and generated
summary files. A pilot packet can demonstrate audit readiness, but final semantic
claims require audit cases sampled from the full eligible matrix.
