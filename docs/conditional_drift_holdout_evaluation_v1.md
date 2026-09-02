# Conditional Drift Temporal Holdout Evaluation v1

## Why this evaluation is different

The 16 event predictions were written before their outcomes were joined. The
evaluation protocol was then persisted as
`conditional-drift-holdout-preregistration-v1-ec634e68b14ac3c9` with:

- the exact 16 event IDs and 14 issuers;
- the frozen O1 challenger and O3 Alignment-only scores;
- a 20-session second-trading-day-open-to-close outcome;
- a pre-acceptance Corwin–Schultz liquidity proxy plus 5 bps, clipped to
  10–100 bps;
- issuer-clustered bootstrap inference with 5,000 fixed-seed replications;
- the full 16-event sample as primary, with support-domain subsets used only as
  sensitivity diagnostics; and
- seven verdict checks that all had to pass.

The preregistration reported `outcome_values_loaded=false`. Only after that
artifact existed did the separate evaluation command open the price windows.

## Result

The Alignment candidate did not validate.

| Diagnostic | O1 challenger | O3 Alignment candidate |
|---|---:|---:|
| 20-session Spearman IC | -0.0735 | -0.1059 |
| MAE | 18.57% | 18.64% |
| Directional accuracy | 43.75% | 37.50% |
| Top-minus-bottom tertile net return | +0.66 pp | -13.50 pp |

The candidate-minus-challenger IC was **-0.0324**. Its issuer-clustered 95%
interval was **[-0.3666, +0.2121]**, with a one-sided bootstrap p-value of
**0.6045**. None of the 14 leave-one-issuer-out samples produced a positive O3
IC. Only the outcome-completeness check passed; the final verdict was
`candidate_effectiveness_not_established` (1/7 checks).

Range-clipping the candidate inputs improved MAE to 17.34% and moved IC to
-0.0559, but this was preregistered as sensitivity-only. It does not rescue the
primary candidate. The seven events fully inside the development feature range
also had a slightly negative IC (-0.0357).

## Tail and endpoint review

Two events had net 20-session market-excess returns above 50%:

- QRHC: issuer price rose from a $1.48 entry open to a $2.27 exit close across
  exactly 20 observed sessions; no split or dividend occurred in the window.
- SND: issuer price rose from a $2.40 entry open to a $3.93 exit close across
  exactly 20 observed sessions; no split or dividend occurred in the window.

Both were non-OTC observations with continuous daily price paths rather than a
single unmatched endpoint. Removing any one issuer still left the Alignment IC
negative, so the verdict is not caused by one tail event.

## Interpretation

The historical development improvement did not generalize to this small
temporal holdout. The result does not prove that every accounting innovation or
post-filing drift effect is absent. It does show that this particular
gross-margin State/Delta/Innovation/Reaction Ridge specification is not ready
for promotion, portfolio backtesting, or live use.

The correct next research move is not to tune the model against these 16 now-
observed outcomes. They become permanent training-only evidence. A new model
version must be motivated by an independent economic hypothesis—most naturally
better expectation data, broader liquidity/state coverage, and explicit cross-
sectional normalization—and must face a newly frozen holdout.

## Reproduction

```powershell
uv run python -m scripts.freeze_conditional_drift_holdout_protocol
uv run python -m scripts.evaluate_conditional_drift_holdout
```

