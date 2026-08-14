# Point-in-Time Price Readiness v1

## Outcome

This round does **not** claim a real-return result. It freezes the contract that a
market-data source must satisfy before SEC disclosure events can enter the
Level/Delta/Innovation return study.

Immutable asset versions:

```text
filing-cover-history-v1-75eb202210a58216
sec-equity-scope-audit-v1-91be9ca9abf87966
massive-reference-audit-v1-c5dd620357f207b1
massive-price-targets-v1-4d7b9b8f154e25a7
massive-price-audit-v1-98657c58e92cab12
exchange-calendar-audit-v1-26f068f384d50b76
sec-price-readiness-v4-e50479210c230b9b
sec-universe-design-v6-5633f0c2703bfe31
```

The frozen 500-unit stage-one sample is unchanged. No unresolved unit was
replaced.

## Legacy cover recovery

The v4 cover parser accepts only visible SEC registration tables headed by
`Title of each class`, optionally `Trading Symbol`, and optionally
`Name of each exchange on which registered`. Narrative ticker mentions are
forbidden because a filing can discuss old, temporary, future, or proposed
symbols and exchanges.

Automatic attribution additionally requires either:

1. a single filing-index subject matching the sampled CIK; or
2. one subject-CIK-matched XBRL ticker and one unambiguous visible title row.

Across the 554 preserved complete covers, the parser found 407 registration rows;
379 were in single-filer subject submissions. A total of 223 accepted security
observations use visible-table evidence, including XBRL-ticker/title joins. This
raises filing-time security identity from 303 to 321 and ticker-plus-exchange
coverage from 199 to 248 among the 1,500 acquired anchors.

Within the fixed probability sample, completed scope decisions increase from
256 to 300:

| Status | Issuer-years |
|---|---:|
| Provisional ordinary common equity | 208 |
| Provisional blank-check equity | 14 |
| Structural SIC 6189 asset-backed | 72 |
| Explicit no trading symbol | 3 |
| Only non-equity titles (after status precedence) | 3 |
| Security identity unresolved | 139 |
| Security title unresolved | 52 |
| Industry unresolved | 9 |

The 222 provisional equity candidates are the current price-mapping workload.
The remaining 200 unresolved scope units must stay in the sample and cannot be
silently deleted.

## Market-data contract

`point_in_time_price_contract.json` requires five linked assets:

1. a stable security identifier plus historical ticker/exchange validity
   intervals;
2. unadjusted daily open and close prices;
3. separately timestamped dividends, splits, and other corporate actions;
4. delisting return, terminal value, or an explicit missing-outcome reason;
5. exchange sessions with UTC regular-open and regular-close timestamps.

Every source response or licensed extract must be preserved with SHA-256.
Historical events may never be joined through the current ticker. The unbiased
entry remains the regular-session open on the second trading session after the
filing acceptance timestamp.

## Provider audit

The capability matrix is deliberately conservative:

- CRSP documents a permanent security identifier (PERMNO), daily open and return
  data, and an explicit delisting total return (`DelRet`). It is the reference
  research candidate, but it is an institutional product and still needs a
  tested timestamped exchange calendar in this project.
- Massive (formerly Polygon.io) documents point-in-time ticker queries, inactive
  and delisted flags, daily OHLC, and separate dividend/split history. Its
  documented holiday endpoint is forward-looking, and a delisted flag/date is
  not a delisting return or terminal payoff.
- Alpha Vantage documents raw daily OHLC and adjusted split/dividend data. Its
  public documentation does not establish the stable historical security master,
  inactive-universe coverage, delisting payoff, or historical session calendar
  required by this contract.

Primary references:

- [CRSP US Stock & Indexes Database Guide](https://www.crsp.org/wp-content/uploads/guides/CRSP_US_Stock_%26_Indexes_Database_Guide_Flat_File_Format_2.0.pdf)
- [CRSP permanent identifiers](https://www.crsp.org/research/permno/)
- [Massive all tickers](https://massive.com/docs/rest/stocks/tickers/all-tickers)
- [Massive day aggregates](https://massive.com/docs/flat-files/stocks/day-aggregates)
- [Massive dividends](https://massive.com/docs/rest/stocks/corporate-actions/dividends)
- [Massive splits](https://massive.com/docs/rest/stocks/corporate-actions/splits)
- [Massive market holidays](https://massive.com/docs/rest/stocks/market-operations/market-holidays)
- [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)

No free or low-cost source is marked as satisfying the complete contract by
itself. This is a documentation result, not a claim that such a source can never
be assembled with supplemental data.

## Current gates

| Gate | Status |
|---|---|
| Stage-one probability sample | Pass: 500 fixed units |
| Cover acquisition | Pass: 500 / 500 |
| Equity scope | Open: 300 / 500 |
| Event-date reference candidate | Pilot: 28 / 34 Basic-window targets |
| Stable security mapping | Open: 0 / 222 intervals validated |
| Daily-price queries | Pilot pass: 28 / 28 collected; 1,035 validated bars |
| Daily prices and actions | Open; unadjusted adapter implemented |
| Delisting outcomes | Open |
| Historical exchange calendar | Pilot pass: 28 / 28; full-sample gate open |
| Real return study | **Blocked by design** |

## Reproduction

```powershell
uv run python -m scripts.build_sec_filing_cover_history --build-only
uv run python -m scripts.build_sec_probability_sample `
  --cover-root data/processed/sec_filing_cover_history/filing-cover-history-v1-75eb202210a58216
uv run python -m scripts.build_massive_price_pilot
uv run python -m scripts.build_exchange_calendar --collect
uv run python -m scripts.build_sec_price_readiness
uv run python -m scripts.build_sec_universe_design
```
