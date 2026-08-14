# Longitudinal effectiveness study v1

## Status

The frozen verdict is **`effectiveness_not_established`**. The study does not
support a claim that gross-margin Innovation is a validated trading signal. It
does show that the point-in-time and survivorship controls are economically
material.

## Frozen question

On the exact same point-in-time complete-case filing events, does standardized
out-of-sample gross-margin Innovation explain 20-session post-filing market
excess total returns better than year-over-year Delta?

The study was pre-registered before the longitudinal market batch was read. The
atomic observation is Company × Filing Event × Information Set. Level, Delta,
raw Innovation, and standardized Innovation are compared only on the same
complete-case event sample.

## Sample and data

- 58 pre-registered 10-K/10-Q filing events from 15 U.S. issuers;
- SEC filing acceptance timestamps and originally reported XBRL facts;
- 49 events with complete representations and 20-session market excess returns;
- 38 10-Q and 11 10-K events in the complete sample;
- 6 events from securities that later delisted, retained in the unbiased sample;
- unadjusted issuer and SPY opens/closes, splits, cash dividends, exchange
  sessions, security identity, ticker transitions, and delisting outcomes.

The complete events run from 2024-10-30 through 2025-08-14 by SEC acceptance
time. This is a technically selected pilot, not a probability sample of the
U.S. equity market.

## Forecast and return contract

The v1 conditional expectation is a pooled expanding-window OLS. Its inputs are
lagged same-quarter gross margin, lagged gross-margin change, lagged revenue
growth, and a lagged as-of industry statistic. Each target forecast uses only
observations available strictly before the target event. Events with the same
availability timestamp are forecast as a batch and cannot train on one another.

Raw Innovation is actual gross margin minus the out-of-sample forecast.
Standardized Innovation divides that error by a standard deviation estimated
only from strictly earlier out-of-sample forecast errors. It is a model-based
accounting innovation, not analyst surprise.

Entry is the unadjusted open of the second future regular exchange session after
filing acceptance. The primary exit is the unadjusted close of the twentieth
holding session. Splits change share count; cash dividends are retained but not
reinvested. The primary outcome subtracts the matched SPY total return.

## Inference

The main statistic is the Spearman information coefficient. The incremental
test compares the Innovation IC with the Delta IC and estimates a rank regression
of return on Delta and Innovation. Confidence intervals and p-values use 10,000
issuer-cluster bootstrap samples with seed `20260814`. Pre-registered robustness
checks include one- and five-session windows, Benjamini-Hochberg adjustment,
leave-one-issuer-out results, ASNS exclusion, 10-Q-only results, quarterly
stability, issuer influence, quantile returns, and a survivorship counterfactual.

## Results

| Test | Estimate | 95% issuer-cluster interval | Bootstrap p |
|---|---:|---:|---:|
| Level IC | -0.153 | [-0.456, +0.214] | — |
| Delta IC | -0.019 | [-0.267, +0.216] | — |
| Raw Innovation IC | +0.045 | [-0.317, +0.359] | — |
| Standardized Innovation IC | +0.045 | [-0.316, +0.356] | — |
| Innovation IC - Delta IC | +0.064 | [-0.274, +0.374] | 0.697 |
| Innovation rank beta, controlling Delta | +0.063 | [-0.376, +0.386] | 0.709 |

The Innovation point estimate is positive, but the interval permits substantial
negative and positive population relationships. Its incremental advantage over
Delta is not statistically established. The three Innovation quantile returns
are -2.89%, +0.92%, and +0.42%, so economic ordering is not monotonic. Excluding
ASNS changes the Innovation IC to -0.053. The quarterly IC changes sign, including
-0.420 in 2025Q2.

The five-session Innovation IC is +0.204, versus -0.014 for Delta. The difference
is +0.219, but its Benjamini-Hochberg adjusted p-value is 0.340. This is a
pre-registered secondary result and can motivate a new independent replication;
it cannot replace the frozen 20-session primary window after outcomes are known.

## Bias and identity findings

The complete sample's mean 20-session market excess return is -0.57%. Deleting
the six later-delisted events changes the mean to +1.24%, an upward shift of
1.81 percentage points. The later-delisted events average -13.54%. This is a
direct measurement of survivorship selection in the pilot; it is not evidence
that Innovation itself is profitable.

One frozen holding window crossed the OB→TEAD ticker transition. Preserved CIK,
provider security id, share-class FIGI, exchange, and boundary-price evidence
supported joining the two symbols. Restoring the omitted negative event increased
the complete sample from 48 to 49 and reduced the Innovation IC from +0.094 to
+0.045. The repair was specified without reading signal values or event returns.

## Interpretation and next gate

The research infrastructure passed an important test: correcting data quality
made the result less favorable rather than silently deleting a negative event.
The signal did not pass the statistical and economic gates required for a
buy/sell model.

The next marginally useful step is an independent sample at least three to five
times the current size, with substantially more issuers, industries, smaller
companies, and delisting or liquidation cases. The current model and sample must
remain frozen. A five-session confirmation can be pre-registered on the new
batch while preserving the 20-session historical comparison. Analyst consensus,
company guidance, filing text, and market-expectation features should be developed
separately and evaluated on an untouched test set.

Run the frozen build after the required local data layers exist:

```powershell
uv run python -m scripts.build_longitudinal_effectiveness
```

The command refuses to report the primary test unless at least 47 of the 58
events pass the complete feature-and-return gate.

## Frozen artifact identifiers

- Pre-registration: `preliminary-effectiveness-preregistration-v1-97df4b07e54937e8`
- Event-return ledger: `longitudinal-event-return-ledger-v1-295f4ff0fa9a202e`
- Final diagnostic: `preliminary-effectiveness-diagnostic-v1-ec29d989f4f879f1`
- Ticker-transition repair: `ticker-transition-repair-v1-34e561bf1d728e67`
