# SEC Historical Universe Pipeline

## Research purpose

This pipeline replaces a present-day ticker list with a historical filing world.
An issuer enters an issuer-year when it filed an original 10-Q or 10-K in that
year. Later delisting, deregistration, acquisition, bankruptcy, ticker change, or
disappearance does not remove its earlier events.

The existing 82-company Global 500 asset remains frozen as the large-established-
issuer benchmark. None of the assets produced here modifies that benchmark.

No return study is run in this pipeline. SEC disclosure facts remain separate
from performance research until point-in-time prices, exchange calendars,
historical security identity, constituents, and delisting returns are available.

## Stage 1: immutable SEC bulk snapshots

The downloader streams official SEC ZIP files, calculates SHA-256 as bytes arrive,
verifies byte counts and ZIP directories, and only then moves the completed file
into a content-addressed path. Distinct responses never overwrite one another.

Snapshots used in this build:

| Source | Bytes | SHA-256 |
|---|---:|---|
| `submissions.zip` | 1,557,481,368 | `b35d0137f6b748b50154781a105bc549859d8d144269ffb5c419428d2ea2653e` |
| `companyfacts.zip` | 1,403,111,507 | `dcebecc7f2d58f71566336ac6cf19d0e1daea1792f3555426b760f6ffcef0590` |

Both files were fetched on 2026-08-13 with the project's identified User-Agent.
The raw ZIPs are excluded from Git because of their size. Their manifests retain
the source URL, response metadata, fetch timestamp, exact byte count, hash, and
local content-addressed path.

## Stage 2: complete 2016–2025 filing catalog

Dataset version:

```text
sec-filing-catalog-v2-e863ecb81e963c7d
```

The SEC bulk ZIP already contained every historical submissions shard required by
the selected window. The builder parsed all 1,757 overlapping shards across 1,164
CIKs, merged them with primary CIK documents, and exact-deduplicated on subject CIK
plus accession. Coverage is complete for this snapshot and window: 1,757 resolved,
zero unresolved. Five exact duplicate subject-accession rows were rejected and
recorded rather than silently retained.

The complete catalog contains 1,057,454 selected filings, 115,528 filing signals,
and 85,336 issuer-years. Of these, 13,106 CIKs and 74,410 issuer-years have an
original 10-Q or 10-K and are population-frame candidates.

| Filing or signal | Events |
|---|---:|
| Original 10-Q | 191,412 |
| Original 10-K | 71,528 |
| 10-Q/A and 10-K/A | 15,349 |
| NT 10-Q and NT 10-K | 25,793 |
| 8-K Item 4.02 non-reliance | 2,517 |
| 8-K Item 4.01 auditor change | 8,584 |
| Formal Form 25 / 25-NSE candidate signals | 19,036 |
| Form 15 reporting deregistration | 8,928 |
| 8-K Item 1.03 bankruptcy/receivership | 1,210 |

Form 25-NSE can be indexed for an exchange as well as an issuer. These counts are
filing-derived candidate signals, not final economic exit labels or unique
delisted securities.

## Identity and time semantics

- `cik` identifies the entity whose submissions history contains the filing.
- `accession_prefix` is accession metadata, not an inferred economic issuer.
- SEC archive URLs use the subject CIK plus accession.
- Filing availability uses `accepted_at`; `filing_date` is not substituted.
- Current ticker, exchange, name, entity type, and SIC fields remain in a separate
  snapshot table and are explicitly forbidden from historical backfill.
- Listing, SEC-reporting, and operating status are three separate state lines.
- Absence of a later comparative filing after exit is censoring, not zero revision.

## Stage 3: Representative and Stress frames

Dataset version:

```text
sec-universe-design-v6-a5292e02dc6c158c
```

The design deliberately separates two samples:

- **Representative candidates:** 74,410 issuer-years across 13,106 CIKs, selected
  only from original periodic-report history. Future exit information is forbidden
  as a sampling feature.
- **Stress membership:** 23,269 CIK-domain memberships across 11,479 CIKs, enriched
  for revision, distress/exit, and corporate-action filings. This frame has no
  population weights and must never estimate population rates.

There are also 47,118 provisional exit-signal events across 9,982 candidate CIKs.
Their final reason remains `unknown` until the underlying security and filing are
resolved.

## Stage 4: accession-matched event-time size

Dataset version:

```text
event-time-size-v1-9920331a12142a9e
```

The size feature uses `us-gaap:Assets` in USD only when all of the following agree:

1. the form is an original 10-Q or 10-K;
2. the Companyfacts accession matches the catalog accession;
3. the fact period end exactly matches the filing report date;
4. the catalog provides the filing acceptance timestamp;
5. the value is finite and positive.

Companyfacts `filed` is retained for audit but is never treated as availability.
The issuer-year anchor is the first accepted original periodic filing with an
eligible Assets fact in that calendar year. Annual asset quartiles are size
strata, not market-cap tiers.

Results:

- 235,146 of 262,940 original periodic filings have a strict Assets fact: 89.43%;
- 62,963 of 74,410 candidate issuer-years have a size anchor: 84.62%;
- annual issuer-year coverage stays between 83.21% and 85.92%;
- 1,795 fact rows were explicitly rejected; no event had conflicting eligible
  Assets values.

An absolute audit screen flags 1,771 issuer-years below USD 10,000 for semantic
review. These may include real shells and very small registrants. The screen does
not delete observations or change quartile membership. No observation exceeds the
separate USD 5 trillion review threshold.

## Stage 5: filing-time identity, security, and SIC

Target and processed versions:

```text
filing-cover-targets-v1-9d9a9ea7c9e40430
filing-cover-history-v1-75eb202210a58216
```

The frozen plan selects the first accepted original periodic filing in each of
the 74,410 candidate issuer-years. The current diagnostic asset preserves 1,500
small filing index pages—150 per year—and 554 complete primary/XBRL covers,
including every unit in the 500-unit probability sample.

- filing-time subject CIK/name/form: 1,500 / 1,500;
- filing-time subject SIC: 1,481 / 1,500;
- subject-matched ticker or explicit no-symbol state: 321 / 1,500 acquired anchors;
- subject-matched ticker and exchange: 248 / 1,500 acquired anchors;
- canonical visible registration rows: 407, of which 379 are single-filer
  subject-attributable.

All 2,219 response manifests point to content-addressed raw bytes. There were no
network acquisition failures. Multi-registrant filings are matched by XBRL
context CIK and by the subject-specific filing-index company block.

The first 1,500 balanced-year observations contain 144 SIC 6189 asset-backed
issuers and 68 SIC 6770 blank-check issuer-years. These are diagnostic counts,
not population rates. They demonstrate that the original periodic-filer frame
is broader than a common-equity company universe.

Full filing documents are large. The project therefore froze a 500-unit
probability sample before acquiring the missing full covers. See
[Filing-time identity and industry history v1](filing_cover_history_v1.md).

## Stage 6: probability sample and equity scope

Plan and scope-audit versions:

```text
sec-probability-sample-plan-v1-bd8b453b0e9e3a28
sec-equity-scope-audit-v1-91be9ca9abf87966
```

The filing frame is stratified by 10 acceptance years and five point-in-time size
states. Ten units per stratum produce 500 selected issuer-years with known
inclusion probabilities and weights that sum to the 74,410-unit frame. Selection
does not use identity, SIC, security, future exit, or parsing outcomes, and no
unit may be replaced after review.

All 500 selected covers are preserved. Strict subject-CIK security-title review
currently finds 208 ordinary equity candidates, 14 blank-check equity candidates,
72 structural asset-backed issuers, and 206 other excluded or unresolved units.
Only 300 scope decisions are complete, so weighted totals remain
diagnostic. See [Historical equity probability sample v1](probability_sample_v1.md).

## Output tables

| Table | Purpose |
|---|---|
| `filing_catalog.parquet` | Accession-level filings with acceptance timestamps |
| `filing_signals.parquet` | Separate exit and revision-stress signal taxonomy |
| `issuer_year_inventory.parquet` | Historical issuer-year frame and event counts |
| `archive_coverage_audit.parquet` | Per-shard coverage and source-member evidence |
| `representative_candidates.parquet` | Candidate issuer-years plus size/gate status |
| `stress_membership.parquet` | Deliberately enriched stress domains |
| `exit_signal_events.parquet` | Provisional exit events requiring resolution |
| `event_time_total_assets.parquet` | Accession-matched filing-event Assets facts |
| `issuer_year_size_anchors.parquet` | First annual event-time Assets anchors |
| `size_coverage_by_year.parquet` | Annual missingness audit |
| `filing_identity_observations.parquet` | Filing-time subject identity evidence |
| `filing_security_observations.parquet` | Subject-CIK-matched ticker/exchange facts |
| `filing_industry_observations.parquet` | Subject-specific filing-time SIC |
| `identity_intervals.parquet` | Identity validity intervals without backfill |
| `industry_intervals.parquet` | SIC validity intervals without backfill |

Every committed Parquet asset has a row count, byte count, and SHA-256 in its
`output_manifest.json`.

## Current freeze gates

| Gate | Status | Meaning |
|---|---|---|
| Historical filing archive | Pass | All overlapping shards in the snapshot were parsed |
| Event-time size | Open | 84.62% available; missingness and flagged tiny values still need resolution |
| Filing-time issuer identity | Partial | 1,500 balanced-year index anchors verified |
| Historical security identity | Open | 321 subject-matched states across 1,500 anchors |
| Historical industry | Partial | 1,481 filing-time SIC anchors; population collection incomplete |
| Stage-one probability sample | Pass | 500 units, 50 strata, weights reconstruct 74,410 |
| Selected cover acquisition | Pass | 500 / 500 selected covers preserved |
| Equity scope | Open | 300 / 500 adjudicated; no replacements allowed |
| Point-in-time price identity | Open | Historical price identifiers are not joined |
| Equity analysis sample frozen | **No** | Stage-one sampling passed; scope and returns remain blocked |

The next gate is to adjudicate the remaining 200 scope units and join stable
point-in-time security identifiers for 222 provisional equity candidates. The
Massive adapter has frozen all 222 targets, including 34 inside its documented
Basic two-year history window. No API key was available during the build, so the
provider audit correctly contains zero collected observations and leaves every
return gate false. See [Massive Point-in-Time Reference Pilot v1](massive_reference_pilot_v1.md)
and [Point-in-Time Price Readiness v1](price_readiness_v1.md).

## Reproduction

```powershell
$env:SEC_USER_AGENT="Your Name or Organization your.email@domain.com"
uv run python -m scripts.build_sec_filing_catalog --start-year 2016 --end-year 2025
uv run python -m scripts.build_sec_size_features
uv run python -m scripts.build_sec_filing_cover_history --artifact-policy index-only --limit 1000
uv run python -m scripts.build_sec_probability_sample
uv run python -m scripts.collect_sec_probability_sample
uv run python -m scripts.build_massive_reference_pilot
uv run python -m scripts.build_sec_price_readiness
uv run python -m scripts.build_sec_universe_design
```

The script defaults point to the immutable snapshot manifests and output versions
used above. Pass explicit manifest/root arguments to rebuild from other preserved
snapshots without overwriting these assets.
