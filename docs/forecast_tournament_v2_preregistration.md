# Forecast tournament v2 preregistration

## Purpose

V2 asks one question before any market return is joined:

> Can a richer point-in-time expectation model forecast gross margin more accurately
> than the prior-year seasonal baseline, out of sample and across time?

The tournament does **not** select a model because its residual is less correlated
with Delta. Representation distinctness is an outcome observed only after the
forecast-accuracy gate is evaluated.

## Frozen evaluation order

Every candidate is evaluated in this order:

1. point-in-time validity;
2. expanding or rolling out-of-sample prediction;
3. forecast accuracy against `GM(i,t-4)`;
4. stability across availability years and companies;
5. calibration and residual diagnostics;
6. only then, Delta–Innovation dependence and rank movement.

No candidate may enter a return study by performing well only on distinctness or
future returns.

## Primary and secondary metrics

The primary metric is MAE improvement against seasonal naive:

```text
MAESkill = 1 - MAE(model) / MAE(seasonal naive)
```

MAE is primary because a small number of extreme gross-margin changes can dominate
RMSE. Secondary metrics are:

```text
RMSESkill = 1 - RMSE(model) / RMSE(seasonal naive)
OOS R2    = 1 - SSE(model) / SSE(seasonal naive)
Bias      = mean(actual - forecast)
```

Paired loss differences are retained event by event. Uncertainty is summarized with
a deterministic two-way company/year clustered bootstrap interval. The interval is
a stability diagnostic, not a p-value target.

## Frozen candidates

| ID | Model | Role |
|---|---|---|
| M0 | `GM(i,t-4)` | Seasonal-naive benchmark |
| M1 | Existing expanding pooled OLS | Frozen v1 benchmark |
| M2 | Company-aware linear model | Uses each company's prior trajectory |
| M3 | Richer PIT ridge model | Adds stable, interpretable PIT features with shrinkage |
| M4 | Nonlinear model | Deferred robustness model; not required for the primary v2 result |

M2 candidate features are fixed conceptually before fitting:

```text
GM(i,t-1)
GM(i,t-4)
GM(i,t-1) - GM(i,t-5)
trailing mean GM known at t-
trailing GM volatility known at t-
prior revenue growth
historically valid industry information
```

M3 may additionally use an as-of peer-reporting feature:

```text
PeerInformation(i,t-) = information from same-period peers
                        whose filings were accepted before company i
```

Permitted components include the number of already reporting peers, their average
same-period margin change, their own seasonal surprises, and days since those peer
reports. A peer filing accepted at or after the target event is forbidden.

## Eligibility gate for downstream return testing

A candidate is eligible only if:

1. every feature cutoff, training end, and forecast generation timestamp strictly
   precedes the target actual;
2. it improves the primary aggregate MAE against M0;
3. primary-metric improvement is positive in a majority of availability years with
   at least 25 forecasts;
4. calibration and residual diagnostics do not reveal a material systematic failure;
5. the clustered uncertainty interval and subgroup loss tables are reported;
6. model choice was made without market returns and without optimizing
   Delta–Innovation correlation.

The gate does not require correlation to fall below an arbitrary threshold.

## Representation diagnostics after model selection

For the selected forecast, report:

```text
ExpectationAdjustment = GM(i,t-4) - Forecast(i,t-)
Innovation             = Delta + ExpectationAdjustment

std(ExpectationAdjustment)
std(Delta)
std(ExpectationAdjustment) / std(Delta)
corr(ExpectationAdjustment, Delta)
median absolute percentile-rank movement
Top and Bottom quintile overlap
share changing quintile
share moving two or more quintiles
```

Both pooled ranks and availability-quarter ranks with at least ten events are shown.
These are representation diagnostics, not trading portfolios.

## Conditional uncertainty

Raw Innovation is the primary v2 representation. Standardized Innovation remains a
secondary specification. The v1 denominator is a global expanding scale of prior OOS
errors, so it is not company- or event-conditional and barely changes ranks.

A future conditional denominator may combine strictly prior company errors, industry
errors, margin volatility, or a valid prediction interval:

```text
Sigma(i,t-) = conditional uncertainty known before event i,t
```

It must pass the same time-ordering tests and cannot use the current residual.

## Hard data gates before returns

The following are mandatory, not optional:

- historical security master, including identifier changes and delistings;
- historical universe membership not conditioned on surviving to 2026;
- historical industry classification or another PIT peer-membership source;
- exchange calendars and explicit filing-time/session mapping;
- PIT prices, benchmarks, corporate actions, and delisting returns;
- frozen reaction and tradable-drift timestamp definitions.

The current `global500_2026` cohort and `provisional_2026_sic_snapshot` can support
engineering and forecast diagnostics only. They cannot support historical alpha
claims.

## Execution record

The frozen common-sample tournament has now been executed.  M4 remains a deferred
robustness model rather than a primary selection candidate.  Exact implementation
parameters that were not numerically specified in this document are identified
separately instead of being presented as externally timestamped preregistration.
See [forecast tournament v2 results](forecast_tournament_v2_results.md).
