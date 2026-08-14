# Forecast tournament v2 results

## Research question

The tournament asks whether a point-in-time expectation model can forecast gross
margin more accurately than the prior-year seasonal baseline before any market
return is joined.  M0-M4 are evaluated on one locked complete-case sample.

## Locked sample

- 812 Company x Filing Event x Information Set observations;
- 71 companies;
- every M0-M4 forecast is available for every scored event;
- training actuals and feature timestamps are strictly earlier than the target
  availability timestamp;
- same-time filings are processed as one non-leaking batch.

One v1 forecast is excluded by the M0-M4 common-sample intersection.  No candidate
receives a larger or easier scoring sample.

## Aggregate result

| ID | Model | MAE skill vs M0 | RMSE skill vs M0 | OOS R2 |
|---|---|---:|---:|---:|
| M0 | Seasonal naive | 0.0% | 0.0% | 0.0% |
| M1 | Frozen pooled OLS v1 | -6.6% | 4.3% | 8.5% |
| M2 | Company-aware OLS | **35.4%** | **36.2%** | **59.3%** |
| M3 | Richer PIT ridge | 33.8% | 35.7% | 58.7% |
| M4 | Nonlinear ridge robustness | 28.9% | 31.5% | 53.0% |

M2 is selected by the frozen primary MAE rule.  M2 improves MAE in all six
availability years containing at least 25 common events.  Its company/year
clustered 95% interval for MAE skill is 21.0% to 46.3%.

M3 also passes the forecast gate but does not beat the simpler M2.  M4 is retained
as the preregistration's deferred nonlinear robustness model and is not a primary
selection candidate; it also does not beat M2.

## What changed relative to v1

M1 treated the same quarter last year as the main company state and added pooled
lagged change, revenue, and industry information.  M2 explicitly adds the company's
most recently disclosed quarter, its previous year-over-year change, trailing mean,
and trailing volatility.  The result says that a company's recent state trajectory
contains substantial information about its next gross-margin level.

This is not evidence that peer complexity or nonlinearity is necessary.  M3's
strictly earlier same-period peer information and M4's nonlinear expansion do not
improve the primary metric beyond M2.

## Selected Innovation representation

For M2:

```text
ExpectationAdjustment = GM(t-4) - ForecastM2(t-)
InnovationM2           = Delta + ExpectationAdjustment
```

On the locked sample, M2 Delta-Innovation Spearman correlation falls to 0.452,
versus 0.873 for M1.  The adjustment's volatility is 81.7% of Delta volatility,
median absolute percentile-rank movement is 12.8 percentage points, 59.4% of
events change quintile, and 25.0% move at least two quintiles.  These facts are
reported only after the accuracy-based selection and were not an optimization
target.

## Calibration and uncertainty

M2's mean error is -0.34 percentage points and its calibration slope is 0.999.
The deterministic two-way pigeonhole bootstrap resamples company and availability
year clusters.  It is a stability interval, not a p-value target.

## Execution-time operationalization

The prose preregistration fixed the candidate roles, feature concepts, primary MAE
metric, evaluation order, and return-data gates.  The following numerical details
were operationalized in code immediately before the first tournament execution and
were not separately timestamped in an external registry:

- M3 ridge penalty: 10;
- M4 ridge penalty: 25;
- M4 transform: clipped training-standardized inputs, squares, and five fixed
  interactions (`GM(t-1) x GM(t-4)`, `GM(t-1) x prior Delta`,
  `GM(t-4) x trailing mean`, `prior revenue growth x lagged industry GM`, and
  `peer report count x peer mean GM`);
- calibration bias limit: the larger of 0.5 percentage points and 25% of M0 MAE;
- acceptable calibration slope: 0.50 to 1.50.

They must remain fixed for replication of this v2 result.  Future changes require a
new version and must not overwrite these outputs.

The same-period peer feature uses the target's calendar period-end quarter and only
peer filings accepted strictly earlier than the target filing.  Peer membership uses
the provisional current SIC2, with current SIC1 fallback, and is therefore valid only
for forecast engineering diagnostics—not historical industry attribution.

## Evidence boundary

Passing the forecast gate does not authorize a return study.  The current cohort is
conditioned on the 2026 Fortune Global 500 and current SEC coverage.  Industry
membership is a provisional current SIC snapshot.  Historical security identities,
historical constituents, historical classifications, exchange calendars, point-in-
time prices, corporate actions, and delisting returns are still absent.

Therefore the valid conclusion is:

> Company-aware point-in-time financial state materially improves gross-margin
> forecasting and produces a more distinct accounting Innovation.  Whether that
> Innovation explains post-filing returns remains untested.
