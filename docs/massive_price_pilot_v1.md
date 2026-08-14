# Massive Unadjusted Daily-Price Pilot v1

## Outcome

This round creates a resumable, content-addressed daily-price adapter for the 28
unique event-date security candidates in the completed Massive Basic reference
window. It does not compute or publish returns.

Build-only immutable assets:

```text
massive-reference-audit-v1-c5dd620357f207b1
massive-price-targets-v1-4d7b9b8f154e25a7
massive-price-audit-v1-98657c58e92cab12
exchange-calendar-audit-v1-26f068f384d50b76
sec-price-readiness-v4-e50479210c230b9b
sec-universe-design-v6-5633f0c2703bfe31
```

The target plan contains 28 filing events: 16 NASDAQ and 12 NYSE candidates.
Every price query is tied to the selected event-date ticker, exchange MIC,
share-class or composite FIGI, reference-response SHA-256, SEC acceptance
timestamp, and accession number. No target is selected or replaced based on
price availability.

The completed authenticated collection contains 1,035 unadjusted daily bars in
28 exact raw snapshots. All 28 targets produce an observed event-window
candidate; no bar was rejected, no target-date pair is duplicated, no price is
non-positive, no provider security ID is missing, and no OTC bar entered the
pilot. Per-target windows contain 34 to 39 bars. These are data-quality results,
not return estimates.

## Query contract

The adapter uses Massive's documented endpoint:

```text
GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
```

It always sends these explicit parameters:

```text
adjusted=false
sort=asc
limit=50000
```

Massive documents that daily aggregates are in Eastern Time, that an interval
with no eligible trades produces no bar, and that `adjusted=false` returns data
not adjusted for splits. The Basic plan currently documents two years of
history. The frozen pilot windows run from 2024-10-30 through 2026-01-16 and are
inside that documented limit as of the 2026-08-14 pilot date.

Primary documentation:

- [Massive Custom Bars (OHLC)](https://massive.com/docs/rest/stocks/aggregates/custom-bars)
- [Massive Splits](https://massive.com/docs/rest/stocks/corporate-actions/splits)
- [Massive Dividends](https://massive.com/docs/rest/stocks/corporate-actions/dividends)

The API key is sent only in the `Authorization: Bearer` header. It is never
written to URLs, logs, manifests, Parquet tables, or failure records. Exact HTTP
response bytes are preserved before parsing, and every parsed bar retains its
source SHA-256.

## Validation

The parser rejects an entire response when:

- the provider does not explicitly report `adjusted=false`;
- the response ticker differs from the event-date ticker;
- status, JSON structure, or pagination is invalid.

Individual bars are rejected when timestamps are duplicated or out of range,
OHLCV values are missing or non-finite, prices are non-positive, volume is
negative, or high/low relationships are internally inconsistent.

The requested window is seven calendar days before the filing date through 45
calendar days after it. This is long enough to observe a candidate second
post-event session open and a candidate 20-session holding-window close under
normal conditions.

## Why the event windows remain candidates

Observed bars do not constitute an authoritative historical exchange calendar.
A halted or illiquid security can have no bar on a valid exchange session.
Therefore the adapter records:

- `candidate_entry_session_date` and unadjusted open;
- `candidate_exit_session_date` and unadjusted close;
- the exact source hash for both endpoints;
- `candidate_only_reason` explaining the missing gates.

It deliberately does not compute a return. These gates remain false:

| Gate | Status |
|---|---|
| Stable historical ticker interval | Open |
| Historical exchange calendar | Pilot pass: 28 / 28; see `exchange_calendar_v1.md` |
| Corporate actions | Open |
| Delisting outcome | Open |
| Pilot daily-price collection | Pass: 28 / 28 |
| Daily-price research readiness | Open pending other contract assets |
| Return study | **Blocked by design** |

## Reproduction

Build from preserved bytes without network access:

```powershell
uv run python -m scripts.build_massive_price_pilot
```

Collect one entitlement probe in the same PowerShell session that contains
`MASSIVE_API_KEY`:

```powershell
uv run python -m scripts.build_massive_price_pilot --collect --limit 1
```

After a successful probe, collect the remaining resumable targets:

```powershell
uv run python -m scripts.build_massive_price_pilot --collect --limit 27
uv run python -m scripts.build_sec_price_readiness
uv run python -m scripts.build_sec_universe_design
```

The collection command skips already completed targets. Authentication or
entitlement failures stop the batch without deleting preserved successes.
