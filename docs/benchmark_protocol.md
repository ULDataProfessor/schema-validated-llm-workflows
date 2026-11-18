# Benchmark Protocol

Reproducible protocol for comparing agent frameworks on business analytics tasks.

## Design

- **Type:** Repeated-measures, multi-framework benchmark  
- **Factors:** Framework (6 levels), Task (5 levels), Trial (3 or more seeds)
- **Controls:** Fixed model (`gpt-4o-mini` default), temperature (0.0), max retries (3), shared system prompts  
- **Unit of analysis:** Single agent run on one input sample  

The benchmark separates structural reliability from task quality. `schema_pass_rate` and `zero_retry_success_rate` measure whether downstream software can consume the output. `quality_score` and `quality_metric` measure the pre-specified semantic outcome for the task.

## Tasks

### Task 1: Structured Data Extraction

**Objective:** Extract typed fields from messy business documents.

**Subtypes:**
- Invoice extraction (`InvoiceExtraction` schema)
- Customer complaint extraction (`ComplaintExtraction` schema)

**Input:** Raw document text (synthetic in the current artifact; optional CFPB subset for robustness extension)  
**Output:** Pydantic-validated JSON matching task schema  
**Ground truth:** Field-level labels in dataset  

**Metrics:**
| Metric | Definition |
|--------|------------|
| `schema_pass_rate` | % runs producing valid Pydantic model |
| `field_macro_accuracy` | Mean exact-match score over normalized fields |
| `retry_count` | Mean validation retries per run |
| `error` / `error_type_counts` | Raw error string in trial rows; classified error taxonomy in summaries |
| `zero_retry_success_rate` | % runs that validate without retry |

**Primary quality metric:** `field_macro_accuracy`

**Prompt templates:** `benchmark/tasks/extraction/prompt_templates/invoice_system.txt` and `benchmark/tasks/extraction/prompt_templates/complaint_system.txt`

---

### Task 2: SQL Query Generation

**Objective:** Generate executable SQL from natural language + schema DDL.

**Input:** Question + `CREATE TABLE` statements (synthetic analytics DB in the current artifact; optional Spider/BIRD-style public extension)  
**Output:** SQL string + optional explanation  
**Ground truth:** Gold SQL or result-set equivalence  

**Metrics:**
| Metric | Definition |
|--------|------------|
| `syntax_valid` | Parses in SQLite without error |
| `execution_success` | Query runs without runtime error |
| `result_correct` | Result set matches gold (order-insensitive) |
| `schema_adherence` | Only uses declared tables/columns |

**Primary quality metric:** `result_correct`

---

### Task 3: Business Report Summarization

**Objective:** Produce structured executive summary from source report.

**Input:** Synthetic quarterly report (tabular + narrative)  
**Output:** `ExecutiveSummary` with required sections  
**Ground truth:** Section checklist + fact keys in source  

**Metrics:**
| Metric | Definition |
|--------|------------|
| `required_fields_present` | Required sections are present and non-empty |
| `source_fact_coverage` | Mean coverage of known synthetic facts: quarter, revenue, growth, named risk, and recommendation |
| `hallucination_rate` | % claims with no source support, used in qualitative review |
| `schema_pass_rate` | Valid `ExecutiveSummary` |

**Primary quality metric:** `source_fact_coverage`

---

### Task 4: Compliance / Policy QA

**Objective:** Answer policy questions with structured citations.

**Input:** Policy document + question  
**Output:** `PolicyAnswer` (answer, citations[], confidence)  
**Ground truth:** Annotated answer spans  

**Metrics:**
| Metric | Definition |
|--------|------------|
| `citation_support` | Cited quote appears in the source document |
| `answer` | Stopword-filtered token F1 against the gold short answer |
| `unsupported_claim_rate` | Claims without valid citation |
| `schema_pass_rate` | Valid `PolicyAnswer` |

**Primary quality metric:** `supported_answer_score`

---

### Task 5: Multi-Step Agent Workflow

**Objective:** Classify request, select tool, extract data, and recommend an action.

**Input:** Synthetic support ticket  
**Output:** `WorkflowResult` (classification, tool_used, extraction, recommendation)  
**Ground truth:** Expected classification, tool, key fields  

**Metrics:**
| Metric | Definition |
|--------|------------|
| `workflow_decision_score` | Mean of classification, tool, and review-decision correctness |
| `tool_used` | Correct tool chosen |
| `failed_runs` | Exceptions or max retries exceeded |
| `trace_steps` | Count of logged agent steps |

**Primary quality metric:** `workflow_decision_score`

---

## Implementation Rules (Fairness)

1. **Same prompts** across frameworks (adapt only binding mechanism, not content).  
2. **Same model and parameters** unless framework cannot support a feature (document exceptions).  
3. **No task-specific prompt tuning** per framework before main runs; DSPy may use dev-set optimization on held-out split only.  
4. **Log everything:** tokens, latency, retries, raw output, validation errors.  
5. **Pin dependency versions** in `benchmark/pyproject.toml`.  
6. **Complexity audit:** LOC, file count, optional dependency count, and framework-specific abstraction markers are counted automatically via `scripts/count_complexity.py`.
7. **Maintainability review:** Human-coded ratings and the schema-change exercise use `docs/maintainability_rubric.md` and `python -m eval.maintainability`; they should be scored before reading benchmark performance results when feasible.
8. **Portable trace export:** Each JSONL trial record is treated as the minimum cross-framework audit trace: framework, task, sample, model, prompt/schema version, raw-output reference, parsed output, validation outcome, retries, latency, token use, task scores, errors, and final disposition where available.
9. **Schema migrations are explicit:** historical records must be validated through a versioned contract, and migrations that cannot reconstruct new evidence fields must route to review rather than fabricating values.
10. **Matrix planning:** Trial counts and approximate maximum Wilson interval widths are generated from `config/trials.yaml` with `scripts/plan_matrix.py` before live runs.
11. **Stress extensions are separate evidence:** repeated-run, perturbation, tool-fault, and schema-drift runs must be labeled separately from the primary matrix so production-robustness claims do not contaminate baseline framework comparisons.
12. **Ablations are mechanism checks:** ablation rows must be labeled with an `ablation_id` from `config/ablations.yaml` and kept separate from the primary framework matrix unless a result table explicitly says it is reporting an ablation.

## Evidence Tiers

| Tier | Required artifact | Claim allowed |
|------|-------------------|---------------|
| Dry-run pipeline | Mock runner traces, summaries, and generated tables | The benchmark software path works without provider calls. |
| Live pilot | Non-dry-run traces for limited samples, seeds, or framework-task subsets | Runner-provider paths can execute and expose implementation issues. |
| Eligible descriptive run | Full configured framework-task-seed matrix with archived raw and parsed outputs | Descriptive comparisons of validity, quality, retries, cost, latency, and errors. |
| Inferential study | Eligible run plus frozen analysis scripts and paired/statistical analysis | Task-specific empirical conclusions and bounded practical recommendations. |
| Ablation study | Labeled rows generated from `config/ablations.yaml` plus paired analysis against the relevant control condition | Mechanism-level claims about provider enforcement, retries, validators, typed dependencies, or model sensitivity. |
| Stress extension | Eligible baseline or pilot rows plus an audited stress manifest and labeled stress rows for repeated calls, perturbations, injected tool faults, or schema drift | Production-robustness evidence under labeled stress conditions, reported separately from the primary matrix. |

Dry runs and pilots remain useful engineering evidence, but limited samples or seeds are not final cross-framework empirical results.

## Ablation Manifest

The ablation manifest in `benchmark/config/ablations.yaml` pre-specifies the mechanism checks used to interpret the primary benchmark. These rows answer why a reliability pattern occurs; they do not create a separate framework ranking.

The current manifest covers five ablation families:

| Ablation | Comparison | Claim boundary |
|----------|------------|----------------|
| Provider-native structured output | Provider JSON-schema mode versus application-layer validation only | Framework claims must account for provider constrained decoding. |
| Retry budget | First-attempt output versus final output after retries | Final validity must be reported with retry, latency, and token burden. |
| Domain validators | Structural schema only versus schema plus task validators | Separate shape validation from business-rule validation. |
| Typed dependencies and tools | Prompt-only context versus typed dependency or tool interfaces | Tie agent-design claims to trace, decision, and recovery evidence. |
| Secondary model subset | Primary model versus the configured secondary model subset | Provider-general claims require secondary-model support. |

Ablation rows should retain the normalized trial-record fields and add at least `ablation_id` and `condition`. Paired comparisons should use the same `(task, sample_id, seed)` keys as the main inferential script whenever both conditions are available.

## Stress Robustness Extension

Stress runs answer a different question from the primary benchmark. The primary matrix asks whether framework choice affects validity, task quality, retry burden, latency, cost, and maintainability under matched clean inputs. Stress runs ask whether the same conclusions survive repeated execution, semantically equivalent input variation, controlled tool/API failures, dynamic recovery from tool anomalies, and schema drift.

The pre-specified manifest is stored in `benchmark/config/stress.yaml` and validated with `python -m eval.stress_manifest`. Use a small manifest entry for each stress condition:

```yaml
stress_id: "sql_timeout_001"
base_sample_id: "sql_014"
task: "sql_generation"
family: "tool_or_api_fault"
manipulation: "force_explain_timeout_once"
expected_invariant:
  result_correct: "same_if_recovered"
  mutation_allowed: false
fault:
  type: "timeout"
  intensity: 1
```

Recommended stress families:

| Family | Examples | Report separately as |
|--------|----------|----------------------|
| Repeated execution | Same sample under same prompt/model/schema across repeated calls | consistency, pass-at-k style reliability, variance in task score |
| Input perturbation | Rephrase business question, reorder policy sections, add irrelevant boilerplate | clean-vs-perturbed quality delta and validation delta |
| Tool or API fault | SQL timeout, rate limit, missing retrieved section, partial tool response, stale but schema-valid tool response | recovery rate, fail-closed rate, retry cause, explicit/implicit and transient/permanent fault class |
| Schema drift | Add a required field, tighten enum, rename tool argument | validation visibility, repair behavior, change locality |

Stress rows should keep the same trial-record fields as ordinary runs and add stress metadata either in row metadata or in an external manifest joined by `(task, sample_id, seed, framework, stress_id)`. Tool-fault rows should distinguish explicit faults that are visible as errors, such as timeouts or rate limits, from implicit semantic faults, such as stale or corrupted tool outputs that still satisfy the tool schema. Do not merge stress rows into primary descriptive or inferential tables unless the table explicitly labels the stress condition.

## Directory Layout

```
benchmark/
|-- config/
|   |-- ablations.yaml       # Pre-specified mechanism checks
|   |-- models.yaml          # Model names, temperature, retries
|   |-- stress.yaml          # Pre-specified stress conditions
|   `-- trials.yaml          # Seeds, sample sizes
|-- data/
|   |-- synthetic/           # Generated JSONL
|   `-- schemas/             # Shared Pydantic models
|-- tasks/
|   |-- extraction/
|   |-- sql_generation/
|   |-- summarization/
|   |-- compliance_qa/
|   `-- multi_step/
|-- frameworks/
|   |-- pydantic_ai/
|   |-- langgraph/
|   |-- dspy/
|   |-- crewai/
|   |-- agno/
|   `-- plain_sdk/
|-- eval/
|   |-- metrics.py
|   `-- report.py
`-- scripts/
    |-- generate_synthetic_data.py
    |-- run_benchmark.py
    |-- plan_matrix.py
    `-- count_complexity.py
```

## Running Experiments

Use Python 3.11-3.13 for the full comparative benchmark. The no-cost dry-run path may work on newer Python releases, but live cross-framework claims require the optional framework packages for every configured condition to be importable.

```bash
# Generate data
python scripts/generate_synthetic_data.py

# Check readiness before provider-backed runs
python scripts/plan_matrix.py
python scripts/plan_matrix.py --pilot
python scripts/check_readiness.py --dry-run
python scripts/check_readiness.py --pilot

# Configured pilot matrix
python scripts/run_benchmark.py --all --pilot --dry-run
python scripts/run_pilot.py --dry-run
python scripts/run_pilot.py

# Full dry run (no API calls)
python scripts/run_benchmark.py --all --dry-run --limit 5
python scripts/run_benchmark.py --all --all-seeds --dry-run --limit 1

# Single framework, single task
python scripts/run_benchmark.py --framework pydantic_ai --task extraction --limit 10
python scripts/run_benchmark.py --framework langgraph --task summarization --limit 10
python scripts/run_benchmark.py --framework dspy --task sql_generation --limit 10
python scripts/run_benchmark.py --framework crewai --task summarization --limit 10
python scripts/run_benchmark.py --framework agno --task multi_step --limit 10
python scripts/run_benchmark.py --framework plain_sdk --task compliance_qa --limit 10

# Full matrix (expensive, checkpointed)
python scripts/run_benchmark.py --all --all-seeds --run-id full-main

# Resume after provider/network interruption
python scripts/run_benchmark.py --all --all-seeds --run-id full-main --resume

# Recompute summaries and render paper tables
python scripts/rescore_trials.py results/<run_id>/trials.jsonl \
  --summary-output results/<run_id>/summary.json --in-place
python -m eval.report results/<run_id>/trials.jsonl \
  --output results/<run_id>/summary.recomputed.json
python -m eval.tables results/<run_id> \
  --output results/<run_id>/tables.tex --allow-pilot
python -m eval.tables results/<run_id> \
  --output results/<run_id>/tables.tex --strict
python -m eval.result_card results/<run_id> \
  --output results/<run_id>/result_card.md --allow-pilot
python -m eval.failure_examples results/<run_id>/trials.jsonl \
  --markdown-output results/<run_id>/failure_examples.md \
  --json-output results/<run_id>/failure_examples.json
python -m eval.human_audit results/<run_id>/trials.jsonl \
  --json-output results/<run_id>/human_audit.json \
  --csv-output results/<run_id>/human_audit.csv \
  --markdown-output results/<run_id>/human_audit.md
python -m eval.human_audit results/<run_id>/human_audit.csv \
  --summarize \
  --summary-output results/<run_id>/human_audit_summary.json \
  --summary-markdown-output results/<run_id>/human_audit_summary.md
python -m eval.judge_triage results/<run_id>/trials.jsonl \
  --json-output results/<run_id>/judge_triage.json \
  --markdown-output results/<run_id>/judge_triage.md
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
python -m eval.ablation_manifest \
  --json-output results/<run_id>/ablation_manifest_audit.json \
  --markdown-output results/<run_id>/ablation_manifest_audit.md \
  --strict
python -m eval.stress_manifest \
  --manifest-output results/<run_id>/stress_manifest.yaml \
  --json-output results/<run_id>/stress_manifest_audit.json \
  --markdown-output results/<run_id>/stress_manifest_audit.md \
  --strict
python -m eval.artifact_audit results/<run_id> \
  --json-output results/<run_id>/artifact_audit.json \
  --markdown-output results/<run_id>/artifact_audit.md \
  --allow-pilot
python -m eval.claim_records results/<run_id> \
  --json-output results/<run_id>/claim_records.json \
  --markdown-output results/<run_id>/claim_records.md \
  --allow-pilot
```

## Output Format

Per-run results are written to `benchmark/results/{run_id}/`:
- `trials.jsonl` - one record per run
- `summary.json` - aggregated metrics, confidence intervals, error counts, and distribution diagnostics
- `meta.json` - run configuration, framework/task lists, seeds, sample limits, dry-run/pilot flags, checkpoint status, planned/completed trial count, command provenance, Python/package environment, git state, and config-file hashes
- `tables.tex` - generated LaTeX result tables after `python -m eval.tables`
- `result_card.md` - generated run documentation after `python -m eval.result_card`
- `failure_examples.md` / `failure_examples.json` - deterministic qualitative examples after `python -m eval.failure_examples`
- `human_audit.md` / `human_audit.csv` / `human_audit.json` - deterministic reviewer packet after `python -m eval.human_audit`
- `human_audit_summary.md` / `human_audit_summary.json` - completed-review summary after reviewer fields are filled
- `judge_triage.md` / `judge_triage.json` - supportive review-routing report after `python -m eval.judge_triage`
- `cost_analysis.md` / `cost_analysis.json` - token-normalized and price-normalized costs after `python -m eval.cost_analysis`
- `cost_projection.md` / `cost_projection.json` - pilot-token projection for a planned target matrix after `python -m eval.cost_projection`
- `ablation_manifest_audit.md` / `ablation_manifest_audit.json` - validation report for the pre-specified mechanism checks after `python -m eval.ablation_manifest`
- `stress_manifest.yaml` / `stress_manifest_audit.md` / `stress_manifest_audit.json` - stress-condition manifest and validation report after `python -m eval.stress_manifest`
- `maintainability.md` / `maintainability.csv` / `maintainability.json` - schema-change exercise worksheet after `python -m eval.maintainability`
- `maintainability_summary.md` / `maintainability_summary.json` - completed maintainability review summary after reviewer fields are filled
- `artifact_audit.md` / `artifact_audit.json` - executable claim-readiness audit after `python -m eval.artifact_audit`
- `claim_records.md` / `claim_records.json` - typed supported/blocked claim records after `python -m eval.claim_records`

The implementation complexity script writes `benchmark/results/complexity.json`.

`meta.json` records the seed list, per-task limits, command provenance, environment snapshot, package versions, git state, and config hashes used for a run. The provenance block stores the start and completion timestamps, working directory, command argv, and serialized CLI arguments. For inclusion in the final empirical paper, a live result directory must match the configured frameworks, tasks, seeds, and sample limits in `config/trials.yaml`. `python -m eval.tables ... --allow-pilot` renders pilot tables with a pilot-only warning, while `python -m eval.tables ... --strict` exits nonzero when final eligibility checks fail.

`run_benchmark.py` appends each completed trial to `trials.jsonl` before starting the next provider call. For long runs, set a stable `--run-id`; if the run is interrupted, rerun the same command with `--resume`. Resume mode validates that the requested model, retry budget, framework list, task list, seeds, dry-run/pilot flags, and task limits match the existing `meta.json`, then skips existing `(framework, task, sample_id, seed)` rows.

Before launching a provider-backed run, `scripts/plan_matrix.py` reports planned per-task rows and approximate maximum Wilson half-widths for binary proportions, while `scripts/check_readiness.py` reports provider-key presence, optional framework imports, generated data counts, configured sample limits, seed count, and planned trial rows. The readiness check exits nonzero if a required key is missing, a configured framework import is unavailable, or a generated dataset has fewer records than the configured limit. `scripts/run_benchmark.py --all --pilot` uses the pilot limits in `config/trials.yaml`, and `scripts/run_pilot.py` wraps the same pilot matrix after running the readiness check.

`scripts/rescore_trials.py` recomputes task-specific scores from stored parsed outputs and current metric code without making provider calls. This supports transparent metric fixes while preserving raw model outputs.

`eval.result_card` writes a Markdown summary for a result directory. The card records eligibility, claim scope, model settings, command provenance, framework and task coverage, seeds, task limits, primary metrics, validation problems, required artifacts, and interpretation limits. A pilot card may be archived as artifact-readiness evidence, but a full empirical result card must be generated with final eligibility checks.

`eval.failure_examples` selects representative qualitative examples from `trials.jsonl` without hand-picking cases. It groups trials into parse or transport failures, schema violations, schema-valid semantic errors, tool or orchestration errors, state or procedure failures, retry exhaustion, and recovered retries. These labels are diagnostic selectors, not verified root-cause judgments; likely layer and remediation should be treated as reviewer hypotheses until adjudicated. Raw model text is omitted by default; use `--include-raw` only when the output is safe to archive or has been redacted.

`eval.human_audit` creates deterministic review packets for semantic calibration. It selects diagnostic rows by task using schema validity, task-aware quality score, low field scores, retries, and error type, then adds passing controls. The packet includes blank reviewer fields for correctness, support, error category, severity, adjudication need, reviewer ID, and notes. Raw model text is omitted by default; use `--include-raw` only for synthetic or redacted traces. With `--summarize`, the same module reads a completed JSON or CSV review file and reports completion, reviewer counts, human-correct and human-supported rates, adjudication flags, severity counts, and pairwise agreement for multiply reviewed cases.

`eval.judge_triage` creates a no-API routing report from archived trial rows. It rejects schema-invalid rows, routes rows with failed evidence/state/procedure checks to review, and can optionally join offline judge labels keyed by `(framework, task, sample_id, seed)`. Judge flags are treated as soft review-prioritization signals. They do not override deterministic evidence checks or completed human-review summaries.

`eval.cost_analysis` estimates cost from archived `tokens_in` and `tokens_out` values. Prices are not hardcoded because provider prices and billing rules change; pass the applicable input and output prices per million tokens at analysis time and archive those values with the generated files. `eval.cost_projection` is a preflight companion for pilot runs: it scales observed per-cell token means to the configured target matrix and marks cells with missing token accounting as unknown. Projection files support budgeting and launch decisions, but final cost claims require `eval.cost_analysis` over the completed eligible run.

`eval.inferential` compares two frameworks only on shared `(task, sample_id, seed)` trial keys. It reports the signed paired mean and median difference, a bootstrap confidence interval for the mean difference, and the number of samples where each framework is better under the requested metric direction. For binary metrics such as `schema_valid` and `zero_retry_success`, it also reports matched success/failure cells: `a_success_b_failure`, `b_success_a_failure`, `both_success`, `both_failure`, and `discordant_pairs`. These fields support McNemar-style paired interpretation or exact matched binary tests without reconstructing the table manually. This script is intended for eligible full-matrix runs; pilot output is useful only as an analysis-pipeline check.

`eval.ablation_manifest` validates `config/ablations.yaml` against the configured task list, expected mechanism-check identifiers, required separation rules, comparison conditions, and known primary metrics. Its generated Markdown and JSON files document that mechanism-level claims about provider enforcement, retries, validators, typed dependencies, and secondary-model sensitivity were pre-specified rather than introduced after reading results.

`eval.stress_manifest` validates `config/stress.yaml` against the configured task list, expected stress families, required separation rules, condition fields, and generated synthetic sample identifiers. Its generated files document stress conditions before robustness rows are run. A ready stress manifest supports only stress-design claims; production-robustness claims still require labeled stress rows and paired clean-versus-stressed analysis.

`eval.artifact_audit` is the executable counterpart to the claim-to-artifact map. It checks a result directory for core files, result-card and table outputs, failure-example files, cost-analysis files, ablation-manifest audit files, stress-manifest files, human-audit packets and completed-review summaries, paired-analysis JSON files, environment metadata, and run provenance. The JSON and Markdown outputs separate software-path, runner-readiness, descriptive-result, inferential, failure-analysis, cost-analysis, mechanism-ablation-manifest, stress-manifest, semantic-human-audit, environment-metadata, and run-provenance readiness. Use `--allow-pilot` for pilot archives, and use `--strict --require descriptive_results --require inferential_claims` only after a full matrix and paired-analysis files are present.

`eval.claim_records` reads the same audit logic and writes a compact list of typed claim records. Each record names a claim tier, whether it is supported or blocked, the weakest evidence tier, required artifacts, missing artifacts, and blocked overclaims. These records are designed for release review and paper drafting: a pilot archive can explicitly support runner-readiness while blocking framework-superiority and production-robustness claims.

## Repository-Level Release Check

The root release check combines the paper and benchmark hygiene gates into one report:

```bash
python scripts/check_release_readiness.py \
  --json-output release_readiness.json \
  --markdown-output release_readiness.md
```

The command runs the benchmark test suite, validates the ablation and stress manifests, builds the paper package, runs the strict paper source checker, scans the LaTeX log, verifies offline bibliography integrity, verifies that `paper/main.pdf` and `paper/build/arxiv-source.tar.gz` exist with plausible sizes, and checks that the cited pilot's generated `claim_records.*` files preserve the expected supported/blocked claim boundary. The bibliography check validates duplicate keys, missing cited entries, required entry metadata, web-source fields, and arXiv identifier/DOI consistency without depending on network access. The generated Markdown and JSON reports also include a claim-scope section that lists supported pilot-level claim IDs and blocked full-matrix or framework-ranking claim IDs. This is a release-readiness check for code, manifests, manuscript packaging, citation hygiene, and cited-pilot claim hygiene. It does not replace `eval.artifact_audit` for a final full-matrix result directory and does not prove that the full live matrix has been completed.

## Trial Record Schema

```json
{
  "trial_id": "uuid",
  "framework": "pydantic_ai",
  "task": "extraction",
  "sample_id": "inv_001",
  "seed": 42,
  "schema_valid": true,
  "field_scores": {"vendor_name": 1.0},
  "field_f1": 1.0,
  "quality_score": 1.0,
  "quality_metric": "field_macro_accuracy",
  "retries": 0,
  "latency_ms": 1200,
  "tokens_in": 450,
  "tokens_out": 120,
  "error": null,
  "raw_output": "...",
  "parsed_output": {"vendor_name": "Acme Corp"}
}
```

## Summary Metrics

`eval.report` recomputes `summary.json` from `trials.jsonl`. Each framework-task group includes:

- `schema_pass_rate` and `schema_pass_ci95`
- `zero_retry_success_rate` and `zero_retry_success_ci95`
- `mean_quality_score`, `quality_metric`, `quality_ci95`, and `field_score_means`
- `mean_retries`, `median_retries`, and `retry_rate`
- `mean_latency_ms`, `median_latency_ms`, and `p95_latency_ms`
- `mean_total_tokens`, `median_total_tokens`, and `p95_total_tokens`
- `error_count`, `error_rate`, and `error_type_counts`

`eval.inferential` supports paired comparisons for `quality_score`, `field_f1`, `schema_valid`, `zero_retry_success`, `latency_ms`, `retries`, `total_tokens`, `tokens_in`, `tokens_out`, and individual field scores using `field:<name>`.
