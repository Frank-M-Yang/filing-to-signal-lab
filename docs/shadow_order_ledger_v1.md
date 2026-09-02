# Shadow order ledger v1

> Historical note: this policy-closed foundation is retained for auditability.
> The current daily pipeline uses the policy-aware v2 described in
> `docs/shadow_order_ledger_v2.md`.

## Purpose

This layer turns the prospective event state machine into an auditable daily
decision record without pretending that a trading policy already exists. Every
frozen company-period receives one decision at the state snapshot's evidence
cutoff. The ledger records why an order is blocked and reserves strict order and
position schemas for the next preregistration.

The module is a paper-research control, not a broker interface or an investment
recommendation.

## Policy-closed contract

Market-Innovation v3 freezes a research signal and a future effectiveness test.
It does **not** freeze a mapping from Innovation to long/short direction, an
entry threshold, position size, portfolio constraint, transaction-cost budget,
or an order lifecycle. Choosing those rules after seeing forward outcomes would
create a new selection path.

For that reason v1 creates no shadow order. Even a clean event that reaches
`innovation_ready` is classified as
`research_signal_ready_policy_closed` until those rules are separately
preregistered. The historical validation convention of entering at the second
future trading-session open and holding 20 sessions is not silently promoted
into a live decision policy.

## Decision classes

| Status | Meaning | What can happen next |
| --- | --- | --- |
| `awaiting_innovation` | Filing/actual/Innovation evidence is incomplete. | Continue point-in-time observation. |
| `quality_review_required` | Consensus composition changed and requires adjudication. | Preserve the event; human review may resolve it. |
| `quality_quarantined` | Provider mapping or coverage evidence is structurally unsafe. | Keep the evidence, but do not admit the event. |
| `terminal_rejected` | The event state machine rejected the event. | No resurrection in this ledger. |
| `research_signal_ready_policy_closed` | A clean Innovation exists, but the trading rule is not preregistered. | Freeze a separate execution and risk protocol first. |

## Artifacts

- `shadow_order_decisions.parquet`: one decision for every frozen event and
  state cutoff;
- `shadow_order_blocks.parquet`: the exact evidence, quality, terminal, or
  policy reason that prevents an order;
- `shadow_orders.parquet`: empty v1 table with the future order schema;
- `shadow_positions.parquet`: empty v1 table with the future position schema;
- `shadow_order_gates.parquet`: coverage, quality, outcome, broker, and live
  authorization controls;
- `shadow_order_ledger_report.json` and `output_manifest.json`: immutable
  lineage and content hashes.

`decision_recorded_at` is inherited from the state machine's `state_as_of`
clock. Rebuilding the same state snapshot therefore produces the same decision
identifiers and dataset version. Runtime time appears only as report metadata.

## Firewalls

- return values are forbidden;
- live authorization is forbidden;
- broker submission is always false;
- target side and requested quantity remain null/zero;
- review and quarantined consensus events cannot become order candidates;
- the intentionally failed `trading_policy_preregistered` gate keeps the paper
  order gate closed while all operational integrity gates can still pass.

Build a ledger directly from an immutable state snapshot:

```powershell
uv run python -m scripts.build_shadow_order_ledger `
  --event-state-root data/processed/prospective_event_state_machine/<version>
```

It is also rebuilt automatically by `scripts.run_forward_research_daily`.
