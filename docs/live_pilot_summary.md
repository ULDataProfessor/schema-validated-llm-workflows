# Live Pilot Summary

Date: 2026-06-12  
Run ID: `20260612T223748237237Z-d2141ca2`  
Environment: Python 3.13.11, `pip install -e "benchmark[all]"` via `uv`  
Model: `openai:gpt-4o-mini`  
Command: `python scripts/run_benchmark.py --all --pilot`

## Scope

The live pilot covered all configured framework and task conditions:

| Dimension | Count |
|---|---:|
| Frameworks | 6 |
| Tasks | 5 |
| Samples per task | 10 |
| Seeds | 1 |
| Trial rows | 300 |

Frameworks: PydanticAI, Plain SDK, LangGraph, DSPy, CrewAI, Agno.  
Tasks: structured extraction, SQL generation, business report summarization, compliance QA, multi-step workflow.

## Result Status

The pilot produced 300 trial records and no recorded runtime errors. Every framework-task cell reached final schema validity of 1.0 in this pilot. The generated pilot tables are available locally at:

`benchmark/results/20260612T223748237237Z-d2141ca2/tables.tex`

The result directory is intentionally under `benchmark/results/`, which is ignored by git. It should be archived separately for any submission that cites pilot data as evidence.

The current local `summary.json` and `tables.tex` were rescored from the stored `parsed_output` records using `scripts/rescore_trials.py` after the summarization metric was updated to handle signed growth percentages. No additional provider calls were made during rescoring.

The generated `artifact_audit.md` marks live runner readiness, failure-mode analysis, cost analysis, and human-audit packet generation as ready for pilot-level claims. It marks descriptive framework-task results, inferential framework comparison, mechanism-ablation claims, stress-robustness claims, maintainability-review claims, completed semantic human-audit claims, environment-metadata claims, and command-provenance claims as not ready. These statuses are intentional: they prevent pilot evidence from being interpreted as final comparative evidence.

The environment and command lines above are contemporaneous pilot notes. The archived `meta.json` predates automatic environment and command-provenance capture, so the artifact audit correctly marks those machine-readable claims as not ready.

## Interpretation

This pilot is artifact-readiness evidence, not final empirical evidence. It demonstrates that the live runner paths, provider requests, task scoring, summary generation, and table rendering work across the full configured framework-task surface under Python 3.13.

It does not support final framework ranking because it uses only one seed and ten samples per task. The full empirical matrix remains six frameworks, five tasks, three seeds, and the configured full task sample limits, for 5,400 planned trial rows.
