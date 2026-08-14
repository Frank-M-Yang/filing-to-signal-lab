# Bias-Aware Backtesting Framework engineering contract

## Product boundary

This repository is an educational research demonstrator. Its purpose is to show
how look-ahead, survivorship, and announcement-timing biases change a backtest—not
to recommend investments or optimize a profitable strategy.

## Research invariants

- The primary research question is whether Level, Delta, or Innovation better
  explains post-filing returns, and how latest/restated data distort that result.
- The atomic unit is Company x Filing Event x Information Set. Compare all
  representations only on the same complete-case event sample.
- Raw and standardized innovations are both first-class outputs. Do not conflate
  Delta with Innovation or describe model-based innovation as analyst surprise.
- Every event forecast must be out of sample. Feature cutoff, training end, and
  forecast generation must all precede the target actual's availability timestamp.
- Forecast sigma uses only strictly earlier out-of-sample forecast errors. Same-time
  event errors, in-sample residuals, and future errors are forbidden.
- The first forecast specification uses lagged industry information. A current-quarter
  industry statistic is forbidden unless it is constructed through a tested as-of join.
- Reaction and tradable-drift windows require explicit timestamps and price types.
  Do not infer them from ambiguous labels such as +1 or +20.
- The unbiased signal uses only originally reported values.
- Restated values are used only when the look-ahead-bias switch is enabled.
- The unbiased universe includes securities that later delist.
- The unbiased execution rule enters at the second trading-day open after the
  announcement. Announcement-close entry is deliberately labelled biased.
- Signal standard deviations use only prior quarters; the current quarter never
  enters its own expectation or volatility estimate.
- Simulations are deterministic for a fixed seed and configuration.
- Calibration parameters are explicit. Do not hide them or describe simulated
  performance as empirical evidence.

## SEC data invariants

- Automated SEC requests must use an identified User-Agent with a real contact
  email and remain below the repository's five-requests-per-second limit.
- Preserve exact response bytes and SHA-256 hashes before parsing. Never overwrite
  a distinct source snapshot with a newer response.
- A companyfacts `filed` date is not a point-in-time availability timestamp. Facts
  without an accession matched to a filing acceptance timestamp are rejected.
- Amendment comparisons require the same taxonomy, tag, unit, period, and base
  form. Missing original facts are labelled `no_prior_match`, never imputed as zero.
- SEC disclosure facts remain separate from performance backtests until point-in-time
  prices, exchange calendars, historical constituents, and delisting returns exist.
- The Apple 2009 case is a retrospective accounting-standard adoption, not a fraud
  or simple error-correction example. Its committed slice must remain traceable to
  the documented accessions, acceptance timestamps, and source-export hash.

## Engineering workflow

Keep the numerical research functions independent from Streamlit. Prefer vectorized
pandas/NumPy transformations. Preserve an adapter interface for later real data.
Public functions require type hints and docstrings.

Before completing work, run:

```text
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```
