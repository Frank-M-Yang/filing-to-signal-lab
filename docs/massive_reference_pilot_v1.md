# Massive Point-in-Time Reference Pilot v1

## Result

This round implements the first authenticated market-data adapter without
opening the return-study gate.

Immutable versions built on 2026-08-14:

```text
massive-reference-targets-v1-5a3375b4bf7d2e90
massive-reference-audit-v1-c5dd620357f207b1
sec-price-readiness-v4-e50479210c230b9b
sec-universe-design-v6-5633f0c2703bfe31
```

The target plan contains all 222 provisional-equity issuer-events in the fixed
probability sample. Selection and acquisition order do not use provider coverage
or query success. No sample unit is replaced.

Massive documents two years of All Tickers history for the free Basic plan and
all history for paid plans. With an explicit 2026-08-14 pilot date and a 730-day
window, 34 targets are inside the documented Basic window: one 2024 event and 33
2025 events. The other 188 remain in the plan and require deeper entitlement or
another source.

The authenticated Basic pilot subsequently collected all 34 in-window targets.
It made 36 requests in total because two active queries returned no rows and
were followed by the required `active=false` query. Exact bytes and hashes were
preserved for all 36 responses.

| Diagnostic | Count |
|---|---:|
| Reference targets | 222 |
| Inside documented Basic history window | 34 |
| Targets collected | 34 |
| Active provider matches | 32 |
| Event-date reference candidates | 28 |
| Validated historical symbol intervals | 0 |
| Final price identities | 0 |
| Return-study eligible | 0 |

The six non-candidates are retained rather than replaced: two provider-no-CIK
matches, two common-stock rows without a stable FIGI (`ALIS` and `MSSA`), one
ambiguous two-class common-stock result (`GTN` and `GTN.A`), and one ADR (`CNTA`)
rejected by the automatic common-stock-only rule.

## Query semantics

For each target, the adapter queries `/v3/reference/tickers` with the sampled
CIK and the SEC filing acceptance calendar date:

1. query `active=true` first;
2. only if that response has zero subject-CIK rows, query `active=false`;
3. preserve every paginated response byte-for-byte before parsing;
4. reject response rows whose CIK or returned active state disagrees with the
   query;
5. never query the current ticker to infer an historical identity.

The API key is sent only in the `Authorization: Bearer` header. It is excluded
from requested URLs, response URLs, logs, raw snapshot manifests, Parquet files,
and error messages. The Basic-pilot command enforces at least 12 seconds between
requests, consistent with Massive's documented five-calls-per-minute Basic
limit.

Official references:

- [All Tickers endpoint](https://massive.com/docs/rest/stocks/tickers/all-tickers)
- [REST authentication](https://massive.com/docs/rest/quickstart)
- [Ticker Events endpoint](https://massive.com/docs/rest/stocks/corporate-actions/ticker-events)
- [Point-in-time ticker-change explanation](https://massive.com/knowledge-base/article/how-does-polygon-handle-ticker-changes-and-acquisitions)

## Matching rule

A provider row becomes only an **event-date reference candidate** when:

- the returned CIK equals the sampled subject CIK;
- it was active on the queried event date;
- the security type is `CS` (common stock);
- the SEC filing-time ticker matches exactly when the SEC ticker exists;
- exactly one row remains;
- a share-class or composite FIGI exists; and
- a four-character primary-exchange MIC exists.

Share-class FIGI is preferred over composite FIGI. A candidate is not yet a
validated validity interval. Experimental ticker-events data is not automatically
concatenated into a price history.

## Why the return gate remains closed

Even a successful reference match supplies only an event-date identity candidate.
It does not supply all of the following:

- tested historical ticker validity boundaries;
- unadjusted entry and exit prices;
- corporate actions over the holding window;
- delisting return or terminal value;
- historical exchange sessions with regular-open and regular-close UTC times.

Consequently `historical_symbol_interval_gate`, `daily_price_gate`,
`delisting_outcome_gate`, `exchange_calendar_gate`, and `return_study_gate` all
remain false by construction.

## Run

Build or inspect the plan without network access:

```powershell
uv run python -m scripts.build_massive_reference_pilot
```

With a deeper-history entitlement, continue the preserved 222-target plan:

```powershell
$env:MASSIVE_API_KEY = (Get-Clipboard).Trim()
Set-Clipboard -Value " "
uv run python -m scripts.build_massive_reference_pilot --collect --limit 5
uv run python -m scripts.build_sec_price_readiness
uv run python -m scripts.build_sec_universe_design
```

The collector is resumable. Valid snapshots are reused and distinct response
bytes never overwrite earlier content.

## Authentication troubleshooting

HTTP 401 means Massive rejected the credential before evaluating historical-data
entitlement. It is different from HTTP 403, which means authentication succeeded
but the plan does not permit the requested query. Massive states that existing
Polygon.io keys remain valid after the rename and that `api.polygon.io` and
`api.massive.com` operate in parallel, so changing domains is not the first fix.

Copy a current key from the Massive dashboard and load it into the same
PowerShell window without printing it:

```powershell
Remove-Item Env:MASSIVE_API_KEY -ErrorAction SilentlyContinue
$env:MASSIVE_API_KEY = (Get-Clipboard).Trim()
"MASSIVE_API_KEY loaded: $(-not [string]::IsNullOrWhiteSpace($env:MASSIVE_API_KEY)); length=$($env:MASSIVE_API_KEY.Length)"
Set-Clipboard -Value " "

uv run python -m scripts.build_massive_reference_pilot --collect --limit 1
```

If this still returns 401, create a new key in the dashboard or contact
`support@massive.com`. Do not paste the key into an issue, chat, screenshot, URL,
or committed `.env` file. Authentication failures do not mark a target complete;
the same target remains pending for a safe retry.
