# Data Science Best Practices

This document defines the required engineering and analytics practices for this repository.

## 1. Repository Standards
- Keep a clear package layout under `src/`.
- Keep executable workflows in `scripts/`.
- Keep tests in `tests/` and documentation in `docs/`.
- Keep artifacts in `artifacts/`.

## 2. Branching and Review
- Use GitFlow-style workflow:
  - `main` for releases.
  - `develop` for integration.
  - `feature/*` for scoped work.
  - `release/*` and `hotfix/*` when needed.
- Merge via PR review.
- Use modular commits with descriptive messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).

## 3. Tooling (Required)
- Dependency management: `uv`.
- Lint/format: `ruff`.
- Type checks: `ty`.
- Unit tests and coverage: `pytest`, `pytest-cov`.
- Load-test scaffold: `locust`.

Use the standard command surface:
- `make setup`
- `make lint`
- `make format`
- `make type`
- `make test`
- `make coverage`

## 4. Configuration and Secrets
- Keep runtime configuration in `.env` (local only).
- Never commit credentials (`config/service_account.json`, API keys, private tokens).
- Keep template examples only (`.env.example`, `config/api_keys_template.env`).

## 5. Data and Artifact Policy
- One-time bootstrap commit of data/artifacts is allowed for team reproducibility.
- After bootstrap, avoid routine commits of raw-data/artifact churn.
- Rebuild artifacts only when feature logic or model behavior changes.

## 6. Analytics and Modeling Practices
- Prefer transparent, auditable baselines first.
- Current baseline uses:
  - metadata text tokens
  - optional Google Vision image labels
  - merged vector representation
  - cosine similarity retrieval
  - optional LightGBM reranking
- Add complexity only when measurable value is demonstrated.

## 7. Testing Expectations
- Add/maintain tests for:
  - data validation
  - feature build logic
  - recommender behavior
  - API endpoints
- Run `make test` before PR.
- Use `make coverage` before release.

## 8. Documentation Expectations
- `README.md` must remain an actionable runbook.
- `docs/METHODOLOGY_AND_DECISIONS.md` must explain technique choices and tradeoffs.
- Update docs when behavior, interfaces, or commands change.
- Documentation sync is mandatory: every relevant code change must include corresponding doc updates in the same PR.

## 9. Runtime Modes
- Consumer mode:
  - uses prebuilt artifacts
  - no external API calls required
- Maintainer mode:
  - rebuilds artifacts
  - requires valid credentials when Vision is enabled

## 10. CI Recommendation
- Minimum CI checks on PRs:
  - lint
  - type
  - test
- Block merges on failing checks.

## 11. Checklist Before Merge
- [ ] Code follows tooling standards (`ruff`, `ty`).
- [ ] Tests added/updated and passing.
- [ ] Docs updated for behavior changes.
- [ ] No secrets in tracked files.
- [ ] Branch and commit hygiene followed.

## References
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [PEP 8](https://pep8.org/)
- [The Twelve-Factor App](https://12factor.net/)
