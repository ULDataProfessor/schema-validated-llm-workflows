# Schema-Validated LLM Workflows

Executable benchmark protocol and comparative evaluation artifact for structured-output integration paths across **PydanticAI**, **LangGraph**, **DSPy**, **CrewAI**, **Agno**, and a framework-free **SDK** reference.

This repository is the public code artifact. The manuscript is published separately on arXiv.

## Repository Layout

```
|-- docs/                    # Protocol and supporting notes
|-- benchmark/               # Reproducible empirical study
|   |-- config/              # Models, trials, ablations, frozen analysis plan
|   |-- data/                # Shared schemas (+ generated synthetic data)
|   |-- tasks/               # Task prompts/templates
|   |-- frameworks/          # Per-framework adapters
|   |-- eval/                # Metrics, reporting, audit helpers
|   |-- scripts/             # Run orchestration
|   `-- tests/               # No-API regression suite
`-- scripts/cloud/           # Optional Arbutus Cloud provisioning helpers
```

## Quick Start

```bash
cd benchmark
# Use Python 3.11-3.13 for the full comparative benchmark.
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env   # add OPENAI_API_KEY for live runs
python scripts/generate_synthetic_data.py
python scripts/plan_matrix.py
python scripts/check_readiness.py --dry-run
python scripts/run_benchmark.py --all --pilot --dry-run
```

Live runs require provider credentials. The eligible confirmatory matrix is `6 × 3 × 300 = 5400` trials:

```bash
python scripts/run_benchmark.py --all --all-seeds --run-id confirmatory-5400-v1
# optional resume helper:
# bash scripts/run_confirmatory_until_done.sh
```

For optional Arbutus Cloud provisioning, see [docs/arbutus_cloud.md](docs/arbutus_cloud.md). Those scripts use your private OpenStack OpenRC session and do not store model-provider keys.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/benchmark_protocol.md](docs/benchmark_protocol.md) | Tasks, metrics, datasets, evaluation |
| [docs/framework_comparison.md](docs/framework_comparison.md) | Implementation-condition comparison notes |
| [docs/human_audit_protocol.md](docs/human_audit_protocol.md) | Blinded human-audit packet protocol |
| [docs/maintainability_rubric.md](docs/maintainability_rubric.md) | Maintainability scoring rubric |
| [docs/threats_to_validity.md](docs/threats_to_validity.md) | Validity threats |
| [docs/live_pilot_summary.md](docs/live_pilot_summary.md) | Historical pilot status notes |
| [benchmark/README.md](benchmark/README.md) | Install and run commands |

## Claim Scope

Reported contrasts are about **system-level structured-output integration paths** under a shared harness, not a general ranking of frameworks or a claim that schema validity equals task correctness. Result directories are local by default (`benchmark/results/` is gitignored); regenerate data and rerun to reproduce the pipeline.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT
