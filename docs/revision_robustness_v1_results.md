# Revision Robustness v1 Results

## Primary result

The walk-forward experiment starts with the locked 812-event M2 common sample.
There are 724 outcomes resolved by a later comparative disclosure or amendment and
88 right-censored outcomes.  Of the resolved outcomes, 49 contain a changed gross-
margin value.

After the frozen minimum-history gate, 503 filing-time decisions are eligible.  Of
those, 417 later have resolved outcomes suitable for evaluating the buffer.

At the primary 95% confidence level:

- median event-time Revision-at-Risk: **0.166 gross-margin percentage points**;
- empirical walk-forward coverage: **96.4%**;
- share of all eligible signals whose direction survives the buffer: **89.7%**;
- resolved signals changing sign after the later value: **4 of 417**;
- robust-classified resolved signals changing sign: **4 of 369**.

The 95% buffer is calibrated in aggregate, but it is not a guarantee for an
individual event.  Rare large redraws can overwhelm it.

## Safety versus actionability

| Confidence | Median buffer | Empirical coverage | Robust decision share | Robust sign flips |
|---:|---:|---:|---:|---:|
| 80.0% | 0.000 pp | 93.3% | 100.0% | 4 |
| 90.0% | 0.000 pp | 93.3% | 100.0% | 4 |
| 95.0% | 0.166 pp | 96.4% | 89.7% | 4 |
| 97.5% | 4.311 pp | 99.3% | 14.1% | 0 |
| 99.0% | 7.440 pp | 99.5% | 5.6% | 0 |

The zero 80% and 90% medians come from the large point mass at no numeric change;
they do not mean revision risk is absent.  The sharp jump above 95% shows a sparse,
heavy tail.  Seeking near-complete protection makes most signals non-actionable.

## Interpretation

The valid result is:

> A pooled 95% point-in-time revision buffer is small for the median event and is
> calibrated on resolved outcomes, but the tail is sufficiently heavy that a
> universal high-confidence safety margin would discard most signals.

This does not establish that a 0.166 pp buffer is an investment rule.  There is no
return evidence, and the pooled distribution does not yet condition on company or
filing characteristics.  The extreme tail includes large accounting-scope or
presentation changes and needs filing-level review.

The next defensible extension is a frozen conditional risk model using only
event-time observables, after historical industry mappings and more verified
revision paths are available.  It should be judged by out-of-sample coverage and
decision stability, never by maximizing backtest returns.
