# Prospective Event State Machine v1

## Purpose

The state machine turns the frozen Market-Innovation v3 registry into an
auditable event lifecycle. It answers four operational questions without
reading returns:

1. What evidence exists for this event now?
2. Which state has the event legitimately reached?
3. Why is it waiting or rejected?
4. Did any event move backwards between immutable daily snapshots?

It does not score a strategy, size a position, place an order, or authorize live
trading.

The state machine consumes the separate consensus-quality audit. All events
remain visible, but only `clean` consensus evidence receives downstream
admission. If raw Innovation can be calculated for a review or quarantined event,
the value remains evidence while the lifecycle stops at `actual_ready` with an
explicit quality blocking reason.

## Implemented lifecycle

```text
registered
  -> consensus_level_ready
  -> consensus_revision_ready
  -> filing_observed
  -> consensus_frozen
  -> actual_review_required (when evidence needs repair)
  -> actual_ready
  -> innovation_ready
```

`price_window_ready`, `paper_decision_ready`, `paper_position_open`, and
`paper_position_closed` are reserved in the state catalog for later modules and
are deliberately unreachable in v1.

An event enters terminal `rejected` when an original filing is observed but no
consensus snapshot exists strictly before SEC acceptance. A rejected event is
never automatically resurrected.

## Two clocks

Every transition preserves both:

- `effective_at`: when the underlying disclosure or fact became effective;
- `observed_at`: when this research pipeline actually possessed the evidence.

`observed_at` must never precede `effective_at`. The distinction prevents a late
collector fetch from being treated as if the model had possessed the data at SEC
acceptance.

## Outputs

- `prospective_event_states.parquet`: one current state per frozen event;
- `prospective_event_state_transitions.parquet`: all reached milestones in order;
- `prospective_event_state_catalog.parquet`: state definitions and reserved states;
- `prospective_event_state_gates.parquet`: coverage, monotonicity, and firewall gates;
- `prospective_event_state_machine_report.json`: state counts and immutable lineage.

The daily runner builds this artifact automatically after the v3 readiness,
actual-revenue, and innovation layers.

## Manual build

```powershell
uv run python -m scripts.build_prospective_event_state_machine `
  --readiness-root <prospective-readiness-root> `
  --actual-audit-root <forward-actual-audit-root> `
  --innovation-ledger-root <forward-innovation-ledger-root> `
  --consensus-quality-root <prospective-consensus-quality-root>
```

## Safety invariants

- The frozen event set cannot change across state snapshots.
- A non-terminal event cannot move to a lower state rank.
- A rejected event cannot automatically return to the active lifecycle.
- Historical backfill and return values are forbidden inputs.
- Review and quarantined consensus evidence cannot enter Innovation state.
- No paper or live position is authorized by this module.
