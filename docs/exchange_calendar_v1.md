# Official NYSE/Nasdaq Exchange Calendar v1

## Outcome

This round replaces price-bar counting with an official historical session
calendar for the 28-event Massive price pilot. All 28 events now have an
explicit regular-session open on the second trading session after SEC filing
acceptance and an explicit regular-session close for the 20-session holding
window. No return is computed.

Immutable versions:

```text
massive-price-audit-v1-98657c58e92cab12
exchange-calendar-audit-v1-26f068f384d50b76
sec-price-readiness-v4-e50479210c230b9b
sec-universe-design-v6-5633f0c2703bfe31
```

## Official evidence

Six official exchange documents were preserved byte-for-byte before calendar
construction:

| Source | Role | SHA-256 prefix |
|---|---|---|
| NYSE 2024 yearly trading calendar | Annual holidays and early closes | `02282e733512` |
| NYSE 2025 yearly trading calendar | Annual holidays and early closes | `3e8b540ad744` |
| Nasdaq 2024 trading calendar | Annual holidays and early closes | `5a417265c706` |
| Nasdaq 2025 trading calendar | Annual holidays and early closes | `fb761082d449` |
| NYSE January 9, 2025 regulatory memo | Exceptional closure | `a4640ce5466c` |
| Nasdaq January 9, 2025 equity alert | Exceptional closure | `50ae14d3328b` |

Primary public pages:

- [NYSE holidays and trading hours](https://www.nyse.com/markets/hours-calendars)
- [Nasdaq trading calendar](https://nasdaqtrader.com/Trader.aspx?id=Calendar)
- [Nasdaq Carter closure alert](https://nasdaqtrader.com/TraderNews.aspx?id=ECA2024-632)

The annual 2025 calendars were prepared before the death of President Jimmy
Carter and do not encode the later January 9, 2025 closure. The pipeline therefore
requires the preserved exceptional-closure documents in addition to the annual
calendars. This is why a weekday-and-holiday formula alone is insufficient for a
research calendar.

## Session construction

The calendar covers 2024-10-30 through 2025-09-28, the union of all 28 price
query windows. Both XNYS and XNAS have 227 identical session dates, producing
454 exchange-session rows.

Regular hours are localized in `America/New_York` and then stored in UTC:

- regular open: 09:30 ET;
- regular close: 16:00 ET;
- official early close: 13:00 ET.

The resulting UTC timestamps correctly change across daylight saving time. Six
early-close rows fall in the pilot range: November 29 and December 24, 2024 and
July 3, 2025 for both exchanges. January 9, 2025 is absent from sessions for both
exchanges and appears as a separately sourced exceptional closure.

## Event-time rule

The alignment compares the full SEC acceptance timestamp with each exchange
session's UTC open and close:

| Acceptance relation | Events |
|---|---:|
| Before regular open | 6 |
| During regular session | 4 |
| After regular close | 17 |
| On a non-session date | 1 |

Only session opens strictly after the filing acceptance timestamp are tradable.
The second such open is the unbiased entry. Starting with that entry session,
the 20th included session supplies the exit close.

This changes six entry dates and six exit dates relative to the earlier
price-bar-only candidate method. Every changed event was accepted before the
regular open. The prior method incorrectly discarded the announcement-day open;
the official calendar moves both endpoints one session earlier. All 28 formal
entry and exit dates have observed unadjusted prices.

## Gates

| Gate | Status |
|---|---|
| Official sources preserved | Pass: 6 / 6 |
| Pilot exchange sessions | Pass: 454 rows |
| Calendar-aligned events | Pass: 28 / 28 |
| Formal endpoint prices observed | Pass: 28 / 28 |
| Stable historical security intervals | Open |
| Corporate actions | Open |
| Delisting outcomes | Open |
| Return study | **Blocked by design** |

`pilot_exchange_calendar_gate=true` applies only to this 28-event pilot.
`exchange_calendar_gate=false` remains correct for the full frozen probability
sample because most historical units do not yet have a validated price identity.

## Reproduction

Collect or reuse the six official files and rebuild the immutable audit:

```powershell
uv run python -m scripts.build_exchange_calendar --collect
uv run python -m scripts.build_sec_price_readiness
uv run python -m scripts.build_sec_universe_design
```

The calendar constructor refuses to build if an official source is missing or
its preserved bytes fail SHA-256 verification.
