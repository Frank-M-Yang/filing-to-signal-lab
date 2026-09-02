# Conditional Drift Forward Candidate v1

## Research purpose

This stage converts the historical Conditional Drift optimization into a
falsifiable holdout experiment. It does **not** add another model after looking
at holdout returns. It freezes two Ridge specifications selected in the prior
development stage:

- `F1_v1_challenger`: the original State + Delta + Innovation + Reaction model;
- `F2_alignment_candidate`: the same model plus the three previously retained
  Innovation–Reaction alignment features.

Both models use the same 35 historical development events, 15 issuers, Ridge
alpha 10, feature order, StandardScaler parameters, coefficients, intercept,
and 20-session liquidity-cost-adjusted target. The last training label became
available at `2025-09-15 20:00:00 UTC`.

## Outcome firewall

The validation registry contains 7,833 events labelled as unobserved. Only
events with a complete filing feature set and an acceptance timestamp strictly
after the training-label cutoff enter the temporal queue. This yields 63
events.

Before scoring, the pipeline reads only event IDs and the `returns_computed`
flag from the local outcome ledger. It does not read return values. Forty-seven
events already have locally materialized outcomes and are therefore quarantined
permanently; they cannot be called unseen validation data later. The remaining
16 clean events all have resolved security identities.

For those 16 events, a separate builder reconstructs only:

- filing-time Level, Delta, and Innovation;
- market state from sessions strictly before SEC acceptance; and
- Reaction through the last completed regular close before the formal second-
  trading-day open.

The resulting feature parquet has no `target_*`, `actual_*`, or return-label
columns. The scorer rejects such columns if they are accidentally joined.

## Predictions recorded before evaluation

All 16 clean events now have immutable challenger and candidate research scores.
No holdout return metric, portfolio P&L, or trading decision has been computed.
This ordering matters: future evaluation can compare previously stored
predictions with separately authorized outcomes without refitting the model.

## Covariate-shift warning

The holdout issuers are materially different from the small development panel.
Only 7 of 16 events lie entirely within the training range on every Alignment
model feature. Two events have severe extrapolation flags (maximum absolute
standardized feature above 4 or more than 20% of features outside the training
range). Raw scores are retained for audit, while range-clipped scores are stored
as a sensitivity diagnostic. The clipping result does not replace the frozen
primary score.

Consequently, the covariate-overlap gate, external-validation gate, and live-
trading gate remain closed. The next legitimate step is to preregister the
evaluation statistics and only then authorize a separate join to the 16 holdout
outcomes.

## Reproduction

```powershell
uv run python -m scripts.run_conditional_drift_forward_freeze
```

The command writes the clean queue and model bundle, builds outcome-free
pre-entry features, and then records frozen predictions. Each stage remains an
independent immutable artifact.
