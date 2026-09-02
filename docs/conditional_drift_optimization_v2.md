# Conditional Drift Optimization v2

Status: frozen historical-development optimization, 2026-08-29.

## Purpose and evidence boundary

V2 attempts to improve Conditional Drift v1 without modifying its frozen output. All 58 source
events and their returns had already been observed, so every v2 metric is explicitly
`historical_development_only`. This round may select a candidate for a future unobserved test; it
cannot validate that candidate.

The v2 atomic unit remains Company × Filing Event × Information Set. SEC acceptance, reaction,
formal entry, exit, and training-label maturity clocks are unchanged from v1.

## Pre-entry feature changes

V2 tests two additions:

1. **Reaction decomposition.** Total pre-entry market-excess reaction is split into the first
   regular-session open strictly after SEC acceptance and the remaining pre-entry reaction. V2
   also adds reaction-range abnormality and strictly pre-event median dollar volume.
2. **Innovation–Reaction alignment.** The model receives the reaction signed by the direction of
   standardized Innovation, a disagreement flag, and the direct Innovation × Reaction
   interaction. A positive signed reaction means that price initially moved in the same direction
   as the financial Innovation.

All reaction features are available before the second-session-open entry. The cost and liquidity
features use 20 sessions ending at the last completed close no later than SEC acceptance.

## Liquidity-aware cost proxy

V1 used fixed 10/25/50 bps round-trip deductions. V2 estimates a pre-event high–low spread using
the overnight-adjusted Corwin–Schultz procedure, adds a frozen five-basis-point slippage allowance,
and caps the base proxy between 10 and 100 bps. Low and stress cases are 0.5× and 2× base, with the
stress case capped at 200 bps.

The estimator follows Corwin and Schultz,
[“A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices”](https://doi.org/10.1111/j.1540-6261.2012.01729.x),
*Journal of Finance* 67 (2012), 719–760, and the authors' overnight-adjusted sample implementation.
This remains a daily-data proxy—not an observed quote, opening-auction impact estimate, borrow
fee, or broker fill.

In the 35 complete events, the median base proxy is 37.19 bps and the 90th percentile reaches the
100 bps cap. The corresponding 20-session same-event mean market-excess return is -1.30%; this is
an unconditional historical-slice description, not model portfolio P&L.

## Fixed development comparison

The immutable output is `conditional-drift-optimization-v2-36838891363182b7`. It retains all 35
v1 complete events. Label-maturity rules permit predictions for 17 events at five sessions and 14
events at 20 sessions.

| Model | Feature change | 5-session IC | 20-session IC |
|---|---|---:|---:|
| O1 | v1 full baseline | +0.576 | +0.182 |
| O2 | decomposed Reaction | +0.368 | -0.029 |
| O3 | v1 + alignment only | **+0.603** | **+0.218** |
| O4 | all v2 features | +0.387 | +0.020 |

The complete v2 feature expansion fails: relative to O1, O4 loses 0.163 of 20-session rank
correlation, raises MAE from 8.55% to 11.81%, and is not promoted. Reaction decomposition is also
discarded from the forward candidate.

The narrow alignment-only model improves rank correlation by +0.027 at five sessions and +0.035
at 20 sessions. Its 20-session leave-one-issuer-out IC remains positive in every exclusion, with a
minimum of +0.022. However, its 20-session MAE is worse than O1 (9.66% versus 8.55%), directional
accuracy is unchanged, and the prediction sample contains only 14 events. The machine-readable
decision is therefore:

`retain_alignment_only_as_low_confidence_external_validation_candidate`

This is a frozen hypothesis for new events, not evidence of alpha. No decomposed/full v2 feature
may be reintroduced for the future test without starting a separately labelled protocol.

## Next test

Only O1 and O3 should be carried into the next outcome-blind expansion. The decision statistic is
the predeclared difference in rank correlation on new events, supplemented by issuer-clustered
uncertainty, MAE, directional accuracy, and cost-aware top-minus-bottom returns. The external
validation and live-trading gates remain closed.

## Reproduce locally

```bash
uv run python -m scripts.build_conditional_drift_v2
```
