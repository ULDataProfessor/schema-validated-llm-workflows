# Threats to Validity

Expected threats for the empirical study, with mitigation strategies.

## Internal Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Implementation bias** | Researcher more familiar with Pydantic/PydanticAI | Blinded complexity review; document implementation hours per framework; consider external contributor for one framework |
| **Prompt adaptation** | Subtle prompt differences when mapping to framework APIs | Single source prompt files; code review checklist; log exact messages sent |
| **Model API differences** | Frameworks may use tool-calling vs. native JSON differently | Record output mode per run; subgroup analysis by mode |
| **Retry policy mismatch** | Unequal retry limits inflate pass rates | Fixed `max_retries=3` across frameworks; log retry counts |
| **Dataset leakage (DSPy)** | Optimizer sees test data | Strict train/dev/test splits; DSPy tunes on dev only |
| **Non-determinism** | LLM stochasticity despite temperature=0 | Multiple seeds; report confidence intervals |

## External Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Model generalization** | Results specific to gpt-4o-mini / claude-sonnet | Primary + secondary model runs on subset |
| **Task scope** | Business analytics != all agent use cases | Explicit scope in paper; discuss creative/open-ended tasks as out of scope |
| **Dataset realism** | Synthetic data may be easier than production mess | Report the current synthetic evidence separately; add CFPB complaints and Spider/BIRD-style SQL as robustness extensions; discuss synthetic bias |
| **Scale** | Benchmark sample sizes may be small | Power analysis for key metrics; report effect sizes |
| **Temporal validity** | Framework APIs change rapidly | Pin versions; archive Docker image; date-stamp runs |
| **Python/package support** | Optional frameworks may not install on the same interpreter version | Use Python 3.11-3.13 for full matrix; readiness check verifies framework imports |

## Construct Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Schema pass != correctness** | Valid JSON with wrong values | Separate structural and semantic metrics; fact-checking for summarization |
| **LOC != maintainability** | Lines of code omits readability | Apply maintainability rubric (1-5); qualitative failure analysis |
| **SQL result match** | Equivalent queries with different SQL | Result-set comparison, not string match |
| **Hallucination detection** | Hard to automate | Rule-based fact keys + sample human audit (n=50) |
| **Latency confounds** | Network variance | Run from fixed region; report p50/p95; same time window |

## Conclusion Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Multiple comparisons** | 6 frameworks x 5 tasks x many metrics | Pre-register primary metrics per task; Bonferroni or FDR correction |
| **Cherry-picking tasks** | Reporting only favorable results | Pre-specify all five tasks in protocol |
| **p-hacking via DSPy** | Extensive hyperparameter search | Cap optimization rounds; report tuning cost |
| **Publication bias** | Negative results for PydanticAI | Commit to reporting all frameworks fairly per critical perspective |

## Ecological Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **No human-in-the-loop** | Production systems use reviewers | Task 5 optional HITL variant in extension |
| **No multi-tenant load** | Benchmark is sequential | Discuss throughput as future work |
| **API cost constraints** | Full matrix may use smaller N | Report pilot N; extrapolate with caution |

## Reproducibility Risks

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **API deprecation** | Models retired | Archive model version strings; provide mock replay mode |
| **Missing secrets** | Cannot rerun without API keys | `--dry-run` and recorded fixtures for CI |
| **Dependency drift** | `pip install` changes behavior | Lock file; supported Python range; Docker compose for benchmark |

## Ethical Considerations

- Use only public or synthetic data; no PII in repository.  
- Report API costs and environmental impact of large trial runs.  
- Avoid implying vendor endorsement; frameworks evaluated critically.
