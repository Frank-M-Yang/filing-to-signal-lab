# Historical equity probability sample v1

## Why this sample exists

The complete filing frame has 74,410 issuer-years, but downloading a full filing
cover for every unit would be expensive and statistically unnecessary. More
importantly, selecting companies because their filings are easy to parse would
create a new coverage bias.

This stage freezes a probability sample before using historical issuer identity,
SIC, ticker, exchange, amendment, exit, or acquisition outcomes.

## Stage-one design

Dataset version:

```text
sec-probability-sample-plan-v1-bd8b453b0e9e3a28
```

The 74,410 issuer-year frame is divided into 50 strata:

```text
10 filing-acceptance calendar years
x
4 point-in-time total-asset quartiles plus an explicit size-missing stratum
```

Ten issuer-years are selected from each stratum, for 500 issuer-years and 488
distinct CIKs. Within a stratum, rank is the frozen SHA-256 sampling hash from the
filing-cover target plan. The design does not use filing-time identity, SIC,
security status, later exit information, parsing success, or eventual data
availability.

For stratum `h`:

```text
pi_h = 10 / N_h
weight_h = 1 / pi_h = N_h / 10
```

Inclusion probabilities range from about 0.59% to 0.94%; weights range from 106.1
to 170.0 issuer-years. The 500 weights sum to 74,410, exactly reconstructing the
filing-frame count apart from floating-point representation.

Units are never replaced after scope review. An asset-backed issuer or missing
security title remains in the selected audit with its original probability and
weight. This prevents a quiet substitution of easy, surviving, or familiar
companies.

## Evidence acquisition

All 500 selected filing index pages were already present in the balanced-year
validation collection. Of the selected units, 446 also had a preserved full
cover. The remaining 54 covers and 17 required instance documents were collected
with 71 additional SEC responses and zero failures.

The full sample now has:

- 500 / 500 filing index artifacts;
- 500 / 500 primary cover artifacts;
- exact raw bytes and content-addressed SHA-256 manifests;
- filing-time subject CIK/name/form and SIC evidence;
- subject-CIK-matched DEI security facts where disclosed.

## Scope protocol

Scope-audit version:

```text
sec-equity-scope-audit-v1-91be9ca9abf87966
```

The adjudication order is frozen:

1. missing filing index or issuer identity remains unresolved;
2. SIC 6189 is structural asset-backed/non-common-equity scope;
3. missing filing-time SIC remains unresolved;
4. missing full cover remains unresolved;
5. an explicit no-symbol state is retained as non-price-ready;
6. only subject-CIK security titles are considered;
7. common stock, ordinary shares, beneficial interests, and common/limited-
   partner units are equity-like only when the title does not also describe a
   warrant, right, preferred security, note, bond, depositary security, or
   combination unit;
8. SIC 6770 equity-like securities remain a separate blank-check stratum;
9. ticker and exchange readiness is a later price-identity gate, not a reason to
   rewrite the stage-one sample.

## Current scope result

| Status | Selected issuer-years |
|---|---:|
| Provisional ordinary common-equity candidate | 208 |
| Provisional blank-check equity candidate | 14 |
| Structural asset-backed/non-common-equity | 72 |
| Explicit no-symbol, not price-ready | 3 |
| Only non-equity security titles observed | 3 |
| Security title missing or unresolved | 52 |
| Security identity unresolved | 139 |
| Filing-time industry unresolved | 9 |

Only 300 of 500 units currently have a completed scope adjudication. Therefore:

- the scope gate is **open**;
- the point-in-time price-identity gate is **open**;
- the return-study gate is **open**;
- no unit is replaced;
- weighted status totals are diagnostics, not final population estimates.

An expanded population total remains premature because 200 selected units remain
unresolved. Reporting a weighted diagnostic as the number or share of public
equities would be misleading.

The size-missing stratum is especially informative: 72 of its 100 sampled units
are SIC 6189 asset-backed issuers, while only three currently qualify as
provisional equity candidates. This is direct evidence that Companyfacts size
missingness is structurally related to issuer type rather than random noise.

## Output assets

| Table | Purpose |
|---|---|
| `selected_issuer_years.parquet` | Frozen sample, probabilities, and weights |
| `sample_filing_cover_targets.parquet` | Exact selected SEC acquisition targets |
| `sampling_strata.parquet` | Population counts, sample counts, probabilities, and weights |
| `sample_scope_audit.parquet` | Every selected unit, including ineligible and unresolved units |
| `scope_by_sampling_stratum.parquet` | Scope completion and weighted diagnostics by stratum |
| `scope_status_summary.parquet` | Unweighted counts and non-final weighted status totals |

## Reproduction

```powershell
uv run python -m scripts.build_sec_probability_sample

$env:SEC_USER_AGENT="Your Name or Organization your.email@domain.com"
uv run python -m scripts.collect_sec_probability_sample

uv run python -m scripts.build_sec_probability_sample `
  --cover-root data/processed/sec_filing_cover_history/filing-cover-history-v1-75eb202210a58216

uv run python -m scripts.build_sec_universe_design
```

The next data task is not model tuning. It is adjudicating the remaining legacy
scope cases and joining a point-in-time security identifier/price source.
Until that work passes, the probability sample is frozen but the equity analysis
sample is not.
