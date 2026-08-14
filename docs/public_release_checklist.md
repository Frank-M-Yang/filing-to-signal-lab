# Public Research Preview release checklist

This checklist defines the repository gate. It is independent of whether any
financial signal is statistically significant.

## Required before the first public push

- [x] Deterministic simulation and public SEC example run without credentials.
- [x] Fresh-clone Streamlit fallback uses aggregate frozen diagnostics only.
- [x] Raw SEC, provider market data, processed datasets, reports, and temporary
  renders are excluded by `.gitignore`.
- [x] No personal contact email or live API credential is present in the public
  candidate set.
- [x] MIT code license and a separate third-party data policy are present.
- [x] README states `effectiveness_not_established` and does not imply a trading
  strategy or validated alpha.
- [x] CI runs formatting, linting, static typing, and all tests.
- [x] A clean export created only from Git-visible files passes the same gates.
- [ ] Repository owner reviews the public candidate file list and chooses the
  GitHub repository name, visibility, and remote.

## Local release audit

```powershell
git status --short
git ls-files --cached --others --exclude-standard
uv sync --locked --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Before staging, confirm that no file under `data/sec/`, `data/market/`,
`data/processed/`, `output/`, `outputs/`, or `tmp/` appears in the Git-visible
file list. Do not use `git add -f` on those paths.

## Suggested first release

- Tag: `v0.5.0-research-preview`
- Scope: framework, Streamlit demo, public fixtures, methods, tests, and frozen
  aggregate results
- Release attachment: the reviewed PDF working paper, uploaded separately from
  the source tree
- Headline: bias controls pass; standalone gross-margin Innovation effectiveness
  is not established

The repository should not publish provider response payloads or claim that the
49-event pilot estimates live trading performance.
