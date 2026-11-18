# Maintainability Rubric

Use this rubric to score each framework implementation after the code is feature complete and before reading framework performance results. Score each dimension from 1 to 5 and record a short note for any score below 4.

## Scoring Dimensions

| Dimension | Score 1 | Score 3 | Score 5 |
|-----------|---------|---------|---------|
| Contract clarity | Output shape is mostly embedded in prompts or parser assumptions. | Shared schema exists, but prompt, runner, and evaluator can drift. | One shared schema drives prompting, validation, serialization, and scoring. |
| Error handling | Failures surface as generic exceptions or invalid text. | Some parse or validation failures are logged, but recovery paths are uneven. | Parse, validation, provider, retry, and tool failures are typed or consistently classified. |
| Traceability | Raw outputs, prompts, or schema versions are missing. | Main artifacts are logged, but retry or tool context is incomplete. | Raw output, parsed output, prompt, schema, model, retries, tools, tokens, and final disposition are retained. |
| Change effort | Adding a field, validator, or schema migration requires changes in many disconnected locations. | Changes are localized but require framework-specific edits that are easy to miss. | Schema, tests, runner, evaluator, and migration changes are localized and easy to review. |
| Testability | Live API calls are required for ordinary regression tests. | Some dry-run or mock tests exist, but failure paths are weakly covered. | Dry-run, mock, and unit tests cover success, validation failure, and missing-dependency behavior. |

## Reviewer Instructions

Reviewers should evaluate the implementation artifact, not model accuracy. When possible, hide framework performance summaries during maintainability scoring. Use the same task scope for every framework: structured extraction, SQL generation, summarization, compliance QA, and multi-step workflow.

The aggregate maintainability score is the mean of the five dimension scores. The paper should retain dimension-level scores because a team may care more about traceability, testability, or change effort than the aggregate.

## Change Exercise

After the initial comparable runners pass dry-run and pilot checks, ask a reviewer to estimate or perform the same change tasks for each framework:

1. Add an evidence object to complaint extraction, including source quote and character span.
2. Add a required `risk_level` enum to compliance answers and route high-risk answers to review.
3. Add a SQL safety validator that rejects mutation statements, undeclared tables, and queries failing `EXPLAIN`.
4. Add an explicit review disposition to the multi-step workflow output.
5. Add a migration path that accepts `v1` complaint records, normalizes them to the current schema, and routes records missing new evidence fields to review.

Record files changed, lines changed, elapsed time, test failures encountered, validation failures before prompt repair, whether old records remain interpretable, and whether prompts, schemas, runners, evaluators, reporting scripts, migration code, and audit records stayed synchronized. The smallest diff is not automatically best; reviewers should score whether the change was localized, understandable, and testable.
