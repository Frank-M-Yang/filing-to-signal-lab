# Forward research daily pipeline v1

The daily runner executes the forward research layers in a fixed order:

1. capture or reuse analyst-consensus snapshots;
2. rebuild Level and cross-date Revision Delta;
3. capture SEC submissions and detect filings accepted after consensus;
4. update every frozen v3 company-period through its SEC acceptance cutoff;
5. collect accession-level companyfacts for detected original filings;
6. build raw and strictly standardized revenue Innovation features;
7. classify provider structure and analyst-coverage contamination;
8. resolve every frozen event through the prospective event state machine;
9. load the outcome-blind, cash-only shadow trading policy;
10. append one policy-aware shadow decision and blocking reason per event;
11. plan acceptance-gated identity, price, split, and dividend evidence;
12. preserve and audit acceptance-gated identity, price, split, and dividend
    responses; and
13. apply cash, gross, issuer, concurrency, and pre-entry clock limits to any
    fully evidenced shadow intention.

The v3 registry-aware layer continues to admit live consensus snapshots after a
fiscal period ends. It stops at the corresponding SEC acceptance timestamp,
because revisions between quarter-end and filing publication are still part of
the market's pre-event information set. One audited submissions request per
issuer is reused across all registered periods.

The state machine records reached milestones, blocking or rejection reasons,
effective and observed clocks, and cross-day non-regression. Later price and
paper-order states are reserved but remain unreachable in v1. See
`docs/prospective_event_state_machine_v1.md`.

The consensus-quality layer keeps every observation but admits only `clean`
events downstream. Review and quarantined events remain visible in the state
ledger and cannot silently become Innovation or paper-trade candidates. See
`docs/prospective_consensus_quality_v1.md`.

The first shadow policy freezes a +/-1% materiality band, cash-only positive
signal actions, 2% event/issuer weights, 20% maximum invested weight, second
session open entry, twentieth-session close exit, and the existing v3 cost
proxy. V2 of the ledger can generate an intention from a future clean
Innovation, but no order exists until official calendar, price, cost, cash, and
portfolio evidence pass. See `docs/shadow_trading_policy_v1.md` and
`docs/shadow_order_ledger_v2.md`.

The market-evidence layer treats current ticker/exchange mappings only as
registration-time candidates. SEC acceptance is required before any query is
planned. Exact response bytes are preserved before parsing; point-in-time
identity, official calendar, 20-session pre-acceptance cost, first-session close
for sizing, and split/dividend evidence must all be observed before the second
session opens. Only then can the USD 100,000 cash-only paper allocator produce
an order candidate. A candidate is never submitted. See
`docs/shadow_market_and_portfolio_v1.md`.

The pipeline downloads market evidence only with `--collect-market`. It does
not fit a model, calculate returns, or submit orders. API credentials are read
from environment variables and are never written to artifacts or logs.

Rebuild from already-preserved data without network access:

```powershell
uv run python -m scripts.run_forward_research_daily
```

Perform a new daily live capture:

```powershell
$env:ALPHA_VANTAGE_API_KEY = "..."
$env:MASSIVE_API_KEY = "..."
$env:SEC_USER_AGENT = "Filing-to-Signal Lab your-real-email@example.com"
uv run python -m scripts.run_forward_research_daily --collect-consensus --collect-sec --collect-market
Remove-Item Env:ALPHA_VANTAGE_API_KEY
Remove-Item Env:MASSIVE_API_KEY
Remove-Item Env:SEC_USER_AGENT
```

The Alpha Vantage interval is at least 12 seconds and SEC access remains below
five requests per second. Re-running a completed capture reuses immutable
completion records rather than spending the same quota again.
