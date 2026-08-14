# Core asset v1: 82-company filing-event cleaning

## Scope

The 82 companies in `gross_margin_ready_cohort.csv` are frozen as the project's
first core research asset. Cleaning status does not change membership: a company
that requires manual review remains in the 82-company registry and is excluded
only from strict model inputs until the review is resolved.

The frozen registry version is:

```text
g500-2026-gm82-2b9bd9467132e3dc
```

Its version is a deterministic hash of the company identity and upstream SEC
dataset versions. Each run also records the source-file SHA-256 and hashes every
output table in `output_manifest.json`.

## Reproducible command

```powershell
uv run python -m scripts.clean_core_asset
```

The run reads existing immutable SEC snapshots and does not make network calls.

## Cleaning layers

### 1. Core membership

`core_asset_82.csv` is the immutable sampling registry. It contains all 82
companies, their Fortune rank, SEC CIK/ticker, upstream dataset version, and the
core asset version.

### 2. Auditable filing facts

An original point-in-time event is created only from an original 10-Q and only
when all of the following hold:

1. the fact period end equals the filing report date;
2. the duration is 70–110 days;
3. revenue and gross profit/cost have the same unit and period start;
4. each selected value resolves uniquely under the fixed XBRL tag priority;
5. the original filing acceptance timestamp is available.

Direct `GrossProfit` is preferred. Otherwise gross profit may be derived as
revenue minus a compatible total cost-of-revenue fact. Scope-sensitive cost tags
are retained for audit but marked for review. Ambiguous facts are rejected, not
averaged or silently imputed.

Every accepted event retains fact IDs, tags, taxonomy, unit, accession, SEC
acceptance time, and source-snapshot hashes. Every rejected filing is retained in
`event_rejections.csv`, and `fact_selection_audit.csv` records the candidate
counts and selection decision.

### 3. Point-in-time Level and Delta

`gm_level` is the original, then-available gross margin. A year-over-year lag is
matched within the same company and fiscal-quarter label using a 330–400 day
period-end gap, choosing the unique candidate closest to 365 days. The lag filing
must have been accepted strictly before the current filing.

`gm_delta_yoy` is available only when that valid prior event exists. A strict
Delta row additionally requires both the current and lag Level rows to pass all
review gates.

### 4. Ex-post comparison

The `latest_*` fields are kept only for the later PIT-versus-latest bias
experiment. Latest revenue and profit/cost must come from one coherent later
filing accession; the cleaner never combines values from different filings.

`revision_source_type` distinguishes an amendment, a later comparative filing,
and the original accession. A later comparative value is not automatically an
SEC amendment or an accounting error. It may reflect presentation changes,
recasts, acquisitions, discontinued operations, or taxonomy changes and must be
interpreted accordingly.

## Frozen run result (2026-08-13)

| Layer | Result |
|---|---:|
| Core companies | 82 |
| Accepted filing events | 2,298 |
| Rejected filing events | 115 |
| Strict Level events | 1,954 |
| Strict Delta events | 1,723 |
| Companies meeting automatic coverage thresholds | 70 |
| Companies retained but requiring manual review | 12 |

The automatic company threshold is at least 16 strict Level events and at least
12 strict Delta events. The 12 review names are UnitedHealth, Cigna, Elevance,
Centene, Kroger, StoneX, Uber, Enbridge, GE Aerospace, Molina, CBRE, and Linde.
Most are quarantined because the currently selected gross-margin concept is not
economically comparable for their industry or because the XBRL cost concept is
scope-sensitive. Enbridge and GE also have insufficient strict coverage after
the review gates.

The accepted table includes 228 events whose ex-post gross margin differs from
the original PIT value. In this snapshot all 228 differences come from later
comparative filings, not amendment filings. This is precisely why the project
must not label every changed XBRL value as a restatement.

## Outputs

| Output | Purpose |
|---|---|
| `core_asset_82.csv` | Frozen 82-company membership registry |
| `quarterly_filing_events.csv` | All accepted events with PIT and ex-post provenance |
| `strict_level_events.csv` | Level-ready model input |
| `strict_delta_events.csv` | Delta-ready model input |
| `event_rejections.csv` | Rejected filings and explicit reasons |
| `fact_selection_audit.csv` | Candidate counts and selection trace |
| `company_quality.csv` | Per-company coverage and review status |
| `research_ready_companies.csv` | The 70-company automatic-ready subset |
| `cleaning_report.json` | Aggregate counts, thresholds, and invariants |
| `output_manifest.json` | SHA-256, byte size, and row count for each CSV |

All outputs live under
`data/processed/global500_2026/core_asset_v1/` and are Git-ignored because they
are reproducible derivatives of large local SEC snapshots.

## Evidence boundary

This cleaning pass establishes a real point-in-time financial-disclosure panel.
It does not establish that Level or Delta predicts returns. Innovation features,
historical industry membership, security lineage, exchange calendars, prices,
benchmarks, delisting returns, and trading costs are still required before a
return experiment can answer the research question.
