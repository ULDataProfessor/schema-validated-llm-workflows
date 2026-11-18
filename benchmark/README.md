# Benchmark Package

Reproducible empirical study comparing agent frameworks on business analytics tasks.

## Install

```bash
# Use Python 3.11-3.13 for the full comparative benchmark.
# Some optional framework packages are not yet available on Python 3.14.
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional framework extras:
pip install -e ".[pydantic_ai]"
pip install -e ".[langgraph,dspy,crewai,agno]"

# Equivalent one-step install for the full comparison:
pip install -e ".[all]"
```

## Commands

```bash
python scripts/generate_synthetic_data.py
python scripts/plan_matrix.py
python scripts/plan_matrix.py --pilot
python scripts/check_readiness.py --dry-run
python scripts/check_readiness.py --pilot
python scripts/run_benchmark.py --all --pilot --dry-run
python scripts/run_pilot.py --dry-run
python scripts/run_benchmark.py --all --dry-run --limit 5
python scripts/run_benchmark.py --all --all-seeds --dry-run --limit 1
python scripts/run_benchmark.py --framework pydantic_ai --task extraction --limit 5
python scripts/run_benchmark.py --framework pydantic_ai --task sql_generation --limit 5
python scripts/run_benchmark.py --framework langgraph --task summarization --limit 5
python scripts/run_benchmark.py --framework dspy --task sql_generation --limit 5
python scripts/run_benchmark.py --framework crewai --task summarization --limit 5
python scripts/run_benchmark.py --framework agno --task multi_step --limit 5
python scripts/run_benchmark.py --framework plain_sdk --task compliance_qa --limit 5
python scripts/run_benchmark.py --all --all-seeds --run-id full-main
python scripts/run_benchmark.py --all --all-seeds --run-id full-main --resume
python scripts/rescore_trials.py results/<run_id>/trials.jsonl --summary-output results/<run_id>/summary.json --in-place
python -m eval.report results/<run_id>/trials.jsonl --output results/<run_id>/summary.recomputed.json
python -m eval.tables results/<run_id> --output results/<run_id>/tables.tex --allow-pilot
python -m eval.tables results/<run_id> --output results/<run_id>/tables.tex --strict
python -m eval.result_card results/<run_id> --output results/<run_id>/result_card.md --allow-pilot
python -m eval.failure_examples results/<run_id>/trials.jsonl \
  --markdown-output results/<run_id>/failure_examples.md \
  --json-output results/<run_id>/failure_examples.json
python -m eval.cost_analysis results/<run_id>/trials.jsonl \
  --input-cost-per-million <input_price> \
  --output-cost-per-million <output_price> \
  --json-output results/<run_id>/cost_analysis.json \
  --markdown-output results/<run_id>/cost_analysis.md
python -m eval.cost_projection results/<pilot_run_id>/trials.jsonl \
  --input-cost-per-million <input_price> \
  --output-cost-per-million <output_price> \
  --json-output results/<pilot_run_id>/cost_projection.json \
  --markdown-output results/<pilot_run_id>/cost_projection.md
python -m eval.inferential results/<run_id>/trials.jsonl \
  --framework-a pydantic_ai --framework-b plain_sdk \
  --task extraction --metric quality_score \
  --output results/<run_id>/paired_pydantic_ai_plain_sdk.json
python -m eval.judge_triage results/<run_id>/trials.jsonl \
  --json-output results/<run_id>/judge_triage.json \
  --markdown-output results/<run_id>/judge_triage.md
python -m eval.ablation_manifest \
  --json-output results/<run_id>/ablation_manifest_audit.json \
  --markdown-output results/<run_id>/ablation_manifest_audit.md \
  --strict
python -m eval.stress_manifest \
  --manifest-output results/<run_id>/stress_manifest.yaml \
  --json-output results/<run_id>/stress_manifest_audit.json \
  --markdown-output results/<run_id>/stress_manifest_audit.md \
  --strict
python -m eval.maintainability \
  --csv-output results/<run_id>/maintainability.csv \
  --json-output results/<run_id>/maintainability.json \
  --markdown-output results/<run_id>/maintainability.md
python -m eval.maintainability results/<run_id>/maintainability.csv \
  --summarize \
  --json-output results/<run_id>/maintainability_summary.json \
  --markdown-output results/<run_id>/maintainability_summary.md
python -m eval.artifact_audit results/<run_id> \
  --json-output results/<run_id>/artifact_audit.json \
  --markdown-output results/<run_id>/artifact_audit.md \
  --allow-pilot
python -m eval.claim_records results/<run_id> \
  --json-output results/<run_id>/claim_records.json \
  --markdown-output results/<run_id>/claim_records.md \
  --allow-pilot
python scripts/count_complexity.py
pytest tests/ -q
```

From the repository root, `python scripts/check_release_readiness.py` runs the benchmark tests, validates the ablation and stress manifests, rebuilds the paper package, checks strict paper source hygiene, scans the LaTeX log, validates offline bibliography integrity, checks the cited pilot's generated claim records, and writes optional release-readiness reports. The Markdown/JSON reports include a claim-scope section showing which pilot-level claims are supported and which framework-ranking or full-matrix claims remain blocked.

`trials.jsonl` is the authoritative artifact for each run. Each row includes task-specific sub-scores plus an explicit `quality_score` and `quality_metric` used for the paper's primary result tables. Live and dry-run records are appended after each completed trial, and `meta.json` is updated with `status`, `planned_trials`, `n_trials`, command provenance, environment metadata, package versions, git state, and config hashes so long runs can be inspected while in progress and reproduced later. `summary.json` is derived from `trials.jsonl` and includes schema-pass rates with Wilson intervals, zero-retry success, task-aware quality, retry rates, p95 latency, token totals, field-level score means, quality metric names, and error-type counts. `eval.tables` validates the result directory against `config/trials.yaml` and renders LaTeX tables for the paper. `eval.result_card` writes a Markdown card documenting run scope, eligibility, claim limits, validation problems, provenance, and primary metrics. `eval.failure_examples` selects deterministic qualitative examples by failure category for manual review and paper excerpts. `eval.cost_analysis` converts archived token counts into cost estimates using explicit per-million-token input/output prices supplied at analysis time. `eval.cost_projection` scales observed pilot token means to the configured target matrix and marks cells with missing token accounting as unknown, so teams can plan paid runs before launching the full matrix. `eval.inferential` computes paired framework differences over shared `(task, sample_id, seed)` keys and bootstraps confidence intervals for the mean difference. `eval.judge_triage` routes rows using programmatic hard stops plus optional offline judge flags; judge labels are supportive review-prioritization signals, not ground truth. `eval.ablation_manifest` validates and documents the pre-specified mechanism checks in `config/ablations.yaml`. `eval.stress_manifest` validates and documents the pre-specified stress-extension conditions in `config/stress.yaml`. `eval.maintainability` generates and summarizes schema-change exercise worksheets. `eval.artifact_audit` writes JSON and Markdown claim-readiness checks for core, inferential, failure-analysis, cost-analysis, ablation-manifest, stress-manifest, human-audit, maintainability, environment metadata, and run provenance. `eval.claim_records` turns that audit into typed supported/blocked claim records for paper claims and release review. Use `--allow-pilot` only for pilot/smoke artifacts; use `--strict` for final full-matrix tables and audits.

Use `--run-id <name>` for long provider-backed runs. If a run is interrupted, repeat the same command with `--resume`; existing `(framework, task, sample_id, seed)` trial keys are skipped and only missing trials are called.

`scripts/plan_matrix.py` reports configured full and pilot matrix sizes, per-task rows, and approximate maximum Wilson interval half-widths. `scripts/check_readiness.py` checks provider-key presence, optional framework imports, generated dataset counts, configured sample limits, selected seed count, and planned trial rows before a live run. Use both before any provider-backed pilot or full matrix run. `scripts/run_benchmark.py --all --pilot` uses the pilot sample limits in `config/trials.yaml`; `scripts/run_pilot.py` is a convenience wrapper that runs the same configured pilot matrix after the readiness check.

`scripts/count_complexity.py` writes `results/complexity.json` with implementation LOC, file count, optional dependency count, and transparent framework-specific abstraction-marker counts.

`scripts/rescore_trials.py` recomputes task scores from stored `parsed_output` values and current metric code without making provider calls. Use it when scoring logic changes but raw model outputs should be preserved.

## Configuration

- `config/models.yaml` - model provider, temperature, retries
- `config/trials.yaml` - seeds, per-task sample limits
- `config/stress.yaml` - pre-specified stress-extension manifest

Set API keys in `.env` (see `.env.example`).
