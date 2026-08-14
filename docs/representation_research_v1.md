# Level / Delta / Innovation research v1

## Research question

This stage asks whether three representations of the same point-in-time gross
margin fact are empirically distinct before any market-return outcome is joined:

```text
Level(i,t)         = GM(i,t)
Delta(i,t)         = GM(i,t) - GM(i,t-4)
RawInnovation(i,t) = GM(i,t) - Forecast(i,t-)
StdInnovation(i,t) = RawInnovation(i,t) / Sigma(i,t-)
```

The atomic observation remains `Company × Filing Event × Information Set`. All
actuals are original 10-Q values with SEC acceptance timestamps. Latest values
are retained only in explicitly labelled hindsight columns.

## Precommitted v1 forecast

The first model is an expanding pooled OLS:

```text
GM(i,t) = beta0
        + beta1 * GM(i,t-4)
        + beta2 * DeltaGM(i,t-4)
        + beta3 * RevenueGrowth(i,t-1)
        + beta4 * IndustryGM(i,t-)
        + error(i,t)
```

Rules fixed before examining market returns:

1. at least 200 strictly earlier complete training events;
2. events with the same SEC acceptance timestamp are predicted as one batch;
3. standardized Innovation requires at least 100 strictly earlier OOS errors;
4. the revenue-growth feature comes from the latest prior company quarter;
5. the industry feature is the median of each peer's latest known PIT margin,
   with at least three peers and a period end at least 45 days before the target;
6. SIC2 is used where possible, with SIC1 as an explicit fallback.

Industry facts are point-in-time, but industry membership currently uses the
2026 SIC snapshot. This is marked `provisional_2026_sic_snapshot` and must be
replaced by historical classifications before a final return study.

## Reproducible command

```powershell
uv run python -m scripts.run_representation_research
```

The run consumes the frozen core-asset cleaning output and does not make network
requests.

## Result (2026-08-13)

Research version:

```text
ldi-pit-v1-35e1162ba3f13789
```

| Stage | Events | Companies |
|---|---:|---:|
| Strict Level | 1,954 | 72 |
| Strict Delta target panel | 1,723 | 71 with eventual common coverage |
| Complete forecast features | 1,013 | — |
| Raw OOS Innovation | 813 | — |
| Standardized common sample | 713 | 71 |
| At least 8 standardized Innovations | — | 65 |

The reductions are expected temporal warm-up, not silent row deletion. Delta
requires a prior-year event; the forecast requires a second layer of lagged
features and 200 earlier training observations; standardization then requires
100 earlier OOS forecast errors.

### Forecast result

Across 813 OOS forecasts:

- pooled OLS RMSE: 6.89 percentage points;
- prior-year seasonal-naive RMSE: 7.14 percentage points;
- RMSE improvement: **+3.5%**;
- pooled OLS MAE: 3.60 percentage points;
- seasonal-naive MAE: 3.37 percentage points;
- MAE improvement: **-6.8%**.

The model therefore produces only a small aggregate RMSE improvement and loses
on MAE. Results also vary materially by year. It is not yet a clearly superior
expectation model.

### Representation distinctness

On the 713-event common sample:

- Spearman correlation between Level and Delta: 0.19;
- Spearman correlation between Delta and Raw Innovation: **0.88**;
- Spearman correlation between Delta and standardized Innovation: **0.88**;
- Spearman correlation between Raw and standardized Innovation: 1.00 after
  rounding.

This is the main v1 finding: Innovation has been constructed without temporal
leakage, but it is not yet sufficiently distinct from Delta. The current model's
conditional expectation remains close to the prior-year margin, so its residual
largely repackages year-over-year change. The appropriate response is to improve
and preregister the expectation model—not to claim a new factor.

The identity behind this result is:

```text
ExpectationAdjustment = GM(i,t-4) - Forecast(i,t-)
Innovation             = Delta + ExpectationAdjustment
```

On the common sample, Expectation Adjustment has a 2.24 percentage-point standard
deviation versus 7.42 percentage points for Delta, a ratio of 30.1%. Its Spearman
correlation with Delta is -0.10. The adjustment is therefore not constant, but it is
small relative to Delta.

The strategy-relevant rank diagnostic adds nuance to the 0.88 correlation:

- median absolute pooled percentile-rank movement: 7.43 percentage points;
- Top-quintile overlap: 76.2%;
- Bottom-quintile overlap: 78.9%;
- events crossing a quintile boundary: 42.9%;
- events moving two or more quintiles: 6.2%.

Innovation does reorder a meaningful number of observations, but mainly across an
adjacent boundary. This is more precise than saying either “the rankings are the
same” or “Innovation is already independent.”

### Forecast stability and uncertainty

The aggregate OOS R2 against seasonal naive is 6.9%. In the six availability years
with at least 25 forecasts, MAE skill is positive in zero years. A deterministic
two-way company/year clustered bootstrap gives these 95% diagnostic intervals:

| Metric | Point estimate | 95% interval |
|---|---:|---:|
| MAE improvement | -6.8% | [-18.6%, +0.8%] |
| RMSE improvement | +3.5% | [-6.2%, +14.2%] |
| OOS R2 | +6.9% | [-12.7%, +26.4%] |

All intervals include zero. V1 therefore fails the frozen accuracy-and-stability
gate and is not eligible for downstream return testing.

### Why Raw and standardized Innovation rank almost identically

The v1 sigma is the standard deviation of all strictly prior OOS errors in a global
expanding history. It is not conditional on company, industry, or event. Across the
713 standardized events, Raw and standardized Innovation have Spearman correlation
0.9967. Standardization remains interpretable as historical error units, but adds
almost no new cross-sectional rank information.

## Revision diagnostic

There are 153 changed-later-value events inside the strict Delta target panel.
The tables show how Level, Delta, and a fixed-PIT-forecast residual would look if
the later actual were substituted. This is a labelled hindsight diagnostic, not
a fully re-estimated latest-data model and not an amendment-frequency estimate.

## Outputs

All generated files live under
`data/processed/global500_2026/representation_research_v1/`.

| Output | Meaning |
|---|---|
| `representation_panel.csv` | All strict Delta targets, feature clocks, forecasts, and provenance |
| `common_representation_sample.csv` | Complete Level/Delta/Raw/standardized Innovation sample |
| `coverage_by_year.csv` | Transparent sample funnel by fiscal year |
| `forecast_diagnostics_by_year.csv` | OOS model vs seasonal-naive error metrics |
| `representation_correlations.csv` | Pearson and Spearman distinctness diagnostics |
| `representation_distributions.csv` | Common-sample distribution summaries |
| `company_coverage.csv` | Per-company representation coverage |
| `revision_diagnostics.csv` | Explicit PIT-versus-later-value hindsight fields |
| `expectation_adjustment_diagnostics.csv` | Identity, scale, rank movement, and quintile overlap |
| `representation_rank_movements.csv` | Event-level pooled and availability-quarter rank changes |
| `forecast_loss_events.csv` | Paired model-versus-naive loss differences |
| `forecast_diagnostics_by_group.csv` | Year, company, SIC, volatility, size-proxy, and filing-lag diagnostics |
| `forecast_bootstrap_intervals.csv` | Company/year clustered forecast-skill intervals |
| `sigma_diagnostics.csv` | Scope and rank effect of the v1 standardization denominator |
| `universe_timepoint_audit.csv` | Explicit current-universe/current-SIC return-study blocker |
| `research_report.json` | Configuration, aggregate results, invariants, and evidence boundary |
| `output_manifest.json` | Row counts, sizes, and SHA-256 hashes |

## Evidence boundary and next gate

This stage does not answer which representation predicts returns. The project
still needs a historical security master, exchange calendars, point-in-time
prices, benchmarks, delisting returns, and explicit reaction/drift windows.

Before joining returns, the v2 expectation tournament follows the frozen rules in
[Forecast tournament v2 preregistration](forecast_tournament_v2_preregistration.md).
Selection uses PIT validity, OOS forecast accuracy, temporal stability, and
calibration. Any reduction in Delta–Innovation dependence is reported afterward as
an outcome; it is never an optimization target.
