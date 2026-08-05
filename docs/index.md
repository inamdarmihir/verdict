# verdict

> Risk classification for agentic coding loops — route high-risk steps to humans before they execute.

__omp_shell("!! danger "The autonomous execution problem"")
    An agent that deletes a production database or pushes a breaking deploy needs a human checkpoint. verdict inserts that checkpoint based on structural risk, not keyword matching.

## What it does

- Structural risk classification of every agent action before execution
- Six risk dimensions: reversibility, blast radius, production signal, auth requirements, data sensitivity, code change scope
- Weighted scoring with per-dimension justification
- Human review checkpoint generation with contextual briefing
- Calibration memory — verdicts improve over time via mem0 outcome tracking
- LangGraph graph: classify → calibrate → exec → record
- Agno agent for verdict management and audit

## Risk dimensions

| Dimension | Weight | High-risk example |
|---|---|---|
| Reversibility | 0.25 | `DROP TABLE`, `git push --force` |
| Blast radius | 0.20 | modifies all users, all environments |
| Production signal | 0.20 | touches `prod`, `main`, live endpoints |
| Auth requirements | 0.15 | requires sudo, service account |
| Data sensitivity | 0.10 | handles PII, credentials, encryption keys |
| Code change scope | 0.10 | >5 files or >200 lines |

See [Quick Start](quickstart.md) to add verdict to your agent loop in under 5 minutes.
