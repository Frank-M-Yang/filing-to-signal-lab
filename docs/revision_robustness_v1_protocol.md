# Revision Robustness v1 Protocol

## Decision question

At an original filing acceptance timestamp, the decision-maker knows the first
reported gross margin and the frozen M2 expectation, but does not know whether a
later filing will redraw the reported value.  This experiment asks:

> How large must the original M2 Innovation be before its direction is robust to a
> historically plausible future numeric revision?

The experiment measures accounting-signal stability only.  It contains no prices,
returns, transaction costs, or investment recommendation.

## Frozen input and representation

- Input: the 812-event M0–M4 locked common sample.
- Expectation model: M2, selected by the preregistered forecast tournament before
  this experiment.
- Initial signal: `M2_innovation_raw = original GM - M2 forecast`.
- Hindsight comparison: `latest innovation = initial innovation + revision_size`.
- M2 forecasts remain frozen.  Latest values never re-enter model training.

## Outcome clock and censoring

A past revision outcome can enter the empirical history only when:

```text
revision_outcome_available_at < target_actual_available_at
```

`revision_outcome_available_at` is the acceptance timestamp of a later comparative
filing or amendment that supplies the latest standardized Companyfacts value.
Same-time outcomes are excluded.

An `original_same_accession` latest fact is **right-censored**, not a confirmed zero
revision.  The current snapshot shows no later fact, but a decision-maker at the
original timestamp could not know that no later fact would arrive.

## Primary buffer

The v1 history scope is pooled gross margin.  Industry conditioning is deferred
because current SIC is a 2026 snapshot and the revision tail is sparse.

For confidence `c`, the event-time buffer is:

```text
RevisionAtRisk(c, t) = empirical higher-quantile c of
                       abs(revision_size)
                       among outcomes resolved strictly before t
```

Primary confidence is 95%.  Sensitivity levels are 80%, 90%, 97.5%, and 99%.
The conservative NumPy `higher` sample quantile is frozen.  A target event becomes
eligible only after at least 100 resolved outcomes and 10 changed outcomes exist.

```text
RobustInnovation = sign(InitialInnovation)
                   * max(abs(InitialInnovation) - RevisionAtRisk, 0)
```

An event is `robust_positive` or `robust_negative` when its absolute initial
Innovation exceeds the buffer.  Otherwise it is `revision_sensitive`.  This is a
probabilistic classification, not a guarantee.

## Diagnostics

The experiment reports:

- empirical coverage of each walk-forward buffer;
- share of decisions whose direction survives the buffer;
- realized sign flips after a resolved later disclosure;
- sign flips that occurred despite a robust classification;
- the confidence-versus-actionability curve;
- source-type and extreme-tail audits.

No confidence level is selected using returns.  The 95% level is the primary
specification; the remaining levels expose the cost of seeking more protection.

## Evidence boundary

`revision_size` means a changed standardized Companyfacts value.  Most observations
come from later comparative filings, not 10-K/A or 10-Q/A amendments.  Changes can
reflect reclassification, presentation, taxonomy, scope, or accounting adoption and
must not automatically be described as errors or fraud.  Extreme events require
filing-level manual review before economic interpretation.
