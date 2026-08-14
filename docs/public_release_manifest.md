# Public Release Manifest

This document defines what belongs in the public research preview and what must stay local.
It is a release boundary, not a claim that every empirical result can be recomputed without
third-party data.

## Included in Git

- Numerical research code, SEC adapters, command-line build scripts, Streamlit application,
  tests, and methodological documentation.
- Small offline fixtures under `data/examples/` and audit manifests under `data/audit/`.
- The curated Apple 2009 filing slice, including accessions, acceptance timestamps, units,
  periods, and source hashes.
- `research_preview_summary.json`, a frozen aggregate-only summary of the 49-event
  longitudinal study. It contains no event-level provider price data.
- Dependency lock file, CI workflow, software license, and third-party data policy.

The exact candidate set can be listed before every release with:

```bash
git ls-files --cached --others --exclude-standard
```

## Excluded from Git

- `.env`, local virtual environments, API keys, SEC identity strings, caches, and temporary
  render directories.
- Raw and processed SEC collections under `data/sec/`, event panels under `data/processed/`,
  and provider data under `data/market/`.
- Generated reports under `output/` and `outputs/`.
- Event-level Massive/Polygon prices and any other third-party payload that is not licensed
  for redistribution.

Placeholder `.gitkeep` files preserve the expected local directory layout without publishing
the contents.

## Reproducibility Levels

1. **Offline recomputation:** the deterministic simulation and Apple filing mechanism can be
   recomputed from the public files.
2. **Offline aggregate verification:** the application and reproduction command verify the
   versioned 49-event summary, but do not reconstruct its event rows.
3. **Credentialed empirical rebuild:** users may run the SEC and market-data pipelines with
   their own contact identity and provider credentials. Outputs remain ignored by Git.

Run the public offline reproduction:

```bash
uv sync --dev
uv run python -m scripts.run_public_reproduction
uv run streamlit run app.py
```

The command writes `output/public_reproduction/public_reproduction_summary.json`. The output
states the evidence type for each layer so a frozen aggregate result cannot be mistaken for an
independently recomputed market study.

## Release Safety Gate

Before pushing, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -m scripts.audit_public_release
uv run pytest
```

The release-audit command checks exactly the files included by a normal `git add .` for private
data roots, absolute user paths, non-placeholder email domains, large artifacts, private-key
files, and common high-confidence token formats. Also inspect staged files with
`git diff --cached --stat` and `git diff --cached`. Commit author metadata is separate from file
contents; use a repository-local GitHub no-reply address if the author does not want a personal
email stored permanently in Git history.
