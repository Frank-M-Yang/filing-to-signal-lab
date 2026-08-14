# 2026 Fortune Global 500 → SEC quarterly-data cohort

## Purpose

This acquisition pass builds a research sampling frame; it does not claim that
the selected companies are representative of the world economy. The source
ranking is the 2026 Fortune Global 500 (fiscal-year 2025 revenue). Financial
statement values are downloaded from SEC EDGAR rather than copied from Fortune.

The workflow starts at rank 1 and proceeds through rank 500. A higher-ranked
company is retained only when it can be mapped conservatively to a current SEC
CIK and has sufficient original quarterly data. Otherwise, the next eligible
company is used as a replacement.

## Reproducible command

```powershell
$env:SEC_USER_AGENT="Your organization your.email@domain.com"
uv run python -m scripts.collect_global500_sec `
  --year 2026 --target 100 --start-year 2016 --end-year 2025
```

The SEC identity is supplied only through the process environment and is not
written to the repository.

## Frozen result (2026-08-13)

- 500 ranking members audited.
- 207 conservative automatic SEC mappings; 49 possible mappings left for manual
  review rather than trusted automatically.
- 100-company primary quarterly-data cohort completed.
- 33 primary-cohort companies came from the original top 100; 67 were
  replacements from ranks 101–500.
- The lowest-ranked primary-cohort replacement was rank 422.
- 82 companies passed the stricter gross-margin research test; 25 were from the
  original top 100 and 57 were replacements.
- 18 additional companies passed the primary data threshold after the first 100
  slots had already been filled.
- 58 mapped registrants lacked supported 10-K/10-Q metadata (usually foreign
  private issuers using 20-F), and 8 had no SEC companyfacts response.

These counts are snapshot-specific and will change when Fortune membership, SEC
registrant identity, company lineage, or source filings change.

## Definitions

Primary quarterly-data completeness requires, during 2016–2025:

1. at least 20 original 10-Q filings with an exact accession-level acceptance
   timestamp; and
2. at least 20 current-quarter revenue events whose period end matches the
   filing report date.

Gross-margin research readiness additionally requires:

1. at least 16 events with revenue plus direct gross profit or a compatible
   total cost-of-revenue fact; and
2. coverage across at least six distinct fiscal years.

The 100-company primary cohort is therefore a data-acquisition cohort. The
82-company table is the current strict starting universe for the
Level-vs-Delta-vs-Innovation experiment. Industry/economic applicability still
needs a second-stage review before estimation.

## Outputs

All generated data are Git-ignored because the SEC snapshots and normalized
tables are large.

| Output | Meaning |
|---|---|
| `data/processed/global500_2026/fortune_global500_ranking.csv` | 500-member rank snapshot and Fortune provenance |
| `data/processed/global500_2026/fortune_to_sec_mapping.csv` | CIK/ticker mapping, method, score, and review status |
| `data/processed/global500_2026/candidate_audit.csv` | Every candidate, failure reason, and coverage metric |
| `data/processed/global500_2026/cohort_100.csv` | Primary 100-company quarterly-data cohort |
| `data/processed/global500_2026/gross_margin_ready_cohort.csv` | Strict 82-company gross-margin subset |
| `data/processed/global500_2026/coverage_report.json` | Machine-readable selection summary and thresholds |
| `data/sec/raw/` | Immutable, content-addressed SEC response bytes |
| `data/sec/normalized/` | Per-CIK/version filings, facts, amendments, rejections, and manifest |

The 82-company registry is now frozen independently from its downstream quality
status. See [core asset cleaning](core_asset_cleaning.md) for event-selection
rules, strict model inputs, output hashes, and the manual-review quarantine.

## Important limitations

- Selection on SEC/10-Q availability creates a strong U.S.-reporting bias. It is
  not a valid sample for claims about all Global 500 companies.
- A current ticker may point to a new holding-company CIK while older filings
  remain under a predecessor CIK. ExxonMobil is one observed lineage case and is
  not automatically spliced across CIKs.
- Banks, insurers, payment networks, airlines, and some service companies do not
  report an economically comparable gross margin. Complete filings do not imply
  that gross margin is a valid state variable.
- Similar company names across countries are unsafe. Name-only fuzzy matches are
  always manual-review candidates; a high string score is not enough.
- This stage contains disclosures only. It does not yet contain point-in-time
  prices, exchange calendars, benchmark returns, or delisting returns, so it is
  not a real return backtest.
