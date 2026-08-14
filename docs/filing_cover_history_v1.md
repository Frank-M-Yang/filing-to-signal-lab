# Filing-time identity and industry history v1

## Question and evidence boundary

The filing catalog identifies SEC submission subjects by CIK, but the bulk
submissions snapshot exposes only current ticker, exchange, name, entity type,
and SIC metadata. Those current fields are not valid historical evidence.

This stage asks a narrower question: what issuer identity, security identity,
and industry information was attached to a specific original 10-Q or 10-K when
the filing was accepted?

The unit remains `subject CIK x filing accession x accepted_at`. No value is
backfilled before its observation. Later delisting or disappearance does not
delete an earlier observation.

## Frozen target plan

Target version:

```text
filing-cover-targets-v1-9d9a9ea7c9e40430
```

The plan contains all 74,410 Representative candidate issuer-years. It selects
the first accepted original 10-Q or 10-K in each subject-CIK calendar year.
Acquisition order cycles through 2016–2025 and uses a SHA-256 rank within year,
so a partial batch is balanced by calendar year rather than concentrated in the
oldest filings or the largest issuers. Future exit information is not used.

## Two evidence tiers

### Small filing index artifact

The accession-specific SEC index page provides a filing-time company block with
subject CIK, registrant name, form, and assigned SIC. It also identifies the
XBRL instance document when one exists.

This is the default population-frame acquisition because the response is usually
only tens of kilobytes.

### Large cover/XBRL artifact

The primary filing document provides DEI facts such as registrant name, CIK,
trading symbol, security title, exchange, and explicit no-symbol flags. For old
non-inline filings, the collector follows the filing index to the XBRL instance
only when the primary document lacks core identity facts.

These documents are much larger. The first 500 complete covers plus their index
pages and 219 required instance documents consumed about 1.7 GiB. Extrapolating
that policy to every candidate would be unnecessarily expensive. Therefore the
pipeline now collects index pages for the population frame first; full cover
documents will be acquired only after a probability sample is frozen.

## Preservation and parsing rules

- Every HTTP response uses the identified project User-Agent and a request
  interval no shorter than 0.20 seconds.
- Exact response bytes are persisted to a content-addressed path before parsing.
- Each target/artifact manifest retains requested URL, final URL, fetch time,
  status, headers, byte count, SHA-256, and local path.
- A distinct response never overwrites earlier bytes.
- Multi-registrant filings are matched by subject CIK. The first DEI CIK on the
  page is never assumed to be the target entity.
- Security facts are accepted for a subject only when their XBRL context entity
  CIK matches that subject.
- Filing index SIC is selected from the subject-CIK company block. A co-filer's
  first SIC is not reused.
- Validity intervals start at `accepted_at` and end immediately before the next
  observation for the same CIK. Nothing is backfilled before the first filing.
- Missing ticker, exchange, or SIC remains missing. Current registry metadata is
  never used as a substitute.

## Current diagnostic acquisition

History version:

```text
filing-cover-history-v1-75eb202210a58216
```

The current local asset contains 1,500 index observations—150 in each year from
2016 through 2025—and 554 complete cover/XBRL observations. These include all 500
units in the frozen probability sample. There were no acquisition failures.

| Evidence result | Issuer-years |
|---|---:|
| Filing index preserved | 1,500 |
| Filing-time issuer CIK/name/form verified | 1,500 |
| Filing-time SIC verified | 1,481 |
| Full cover/XBRL preserved | 554 |
| Subject-matched ticker or explicit no-symbol state | 321 |
| Subject-matched ticker and exchange | 248 |

The issuer identity result is not the security result. An issuer can be perfectly
identified while its common-equity security mapping remains unresolved.

## Population-frame contamination discovered

The first 1,500 balanced-year index observations include:

- 144 issuer-years with filing-time SIC 6189, Asset-Backed Securities;
- 68 issuer-years with filing-time SIC 6770, Blank Checks;
- 19 issuer-years whose filing company block has no SIC.

These are diagnostic sample counts, not unweighted population rates. They show
that “filed a 10-Q or 10-K” is too broad to mean “listed operating common-equity
company.” Asset-backed issuers must not be relabelled as delisted companies, and
blank-check companies need an explicit special-purpose stratum rather than silent
deletion.

## Output assets

| Table | Purpose |
|---|---|
| `filing_cover_targets.parquet` | Frozen 74,410-target acquisition order |
| `artifact_snapshots.parquet` | Exact response lineage and hashes |
| `filing_cover_facts.parquet` | Long DEI cover facts with context entity CIK |
| `filing_visible_security_evidence.parquet` | Canonical visible registration-table rows |
| `filing_identity_observations.parquet` | Filing-time subject identity evidence |
| `filing_security_observations.parquet` | Context-matched security facts |
| `filing_industry_observations.parquet` | Subject-specific filing SIC |
| `issuer_year_identity_industry_anchors.parquet` | Complete target frame and gate flags |
| `identity_intervals.parquet` | Non-backfilled issuer identity intervals |
| `security_intervals.parquet` | Non-backfilled observed security intervals |
| `industry_intervals.parquet` | Non-backfilled SIC intervals |
| `identity_industry_coverage_by_year.parquet` | Total and acquired-conditional coverage |
| `filing_cover_rejections.parquet` | Explicit missing/unsupported evidence records |

Every processed artifact has a row count, byte count, and SHA-256 in the output
manifest.

## Reproduction and continuation

```powershell
$env:SEC_USER_AGENT="Your Name or Organization your.email@domain.com"

# Continue the small population-frame index collection.
uv run python -m scripts.build_sec_filing_cover_history `
  --artifact-policy index-only --limit 1000

# Rebuild processed tables without making any network request.
uv run python -m scripts.build_sec_filing_cover_history --build-only

# Full cover collection is deliberately explicit and should be used only for a
# frozen sample or a bounded parser-validation batch.
uv run python -m scripts.build_sec_filing_cover_history `
  --artifact-policy index-and-cover --limit 100
```

The 500-unit stage-one probability sample is now frozen and its covers have been
collected. The next gate is to resolve legacy security titles inside that fixed
sample and join point-in-time security identifiers and prices. See
[Historical equity probability sample v1](probability_sample_v1.md).
