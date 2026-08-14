# Data and artifact policy

The MIT license in this repository applies to original project code and
documentation. It does not grant rights to third-party source data.

## Included public fixtures

Files under `data/examples/` are small, source-attributed research fixtures used
to explain and test the point-in-time methodology. Their metadata preserves the
relevant SEC source accessions and hashes. They are not a market-wide dataset
and must not be presented as investment evidence.

`research_preview_summary.json` contains aggregate, frozen diagnostics used by
the public Streamlit fallback. It does not contain event-level filings, prices,
security identifiers, or provider response payloads.

## Excluded local data

The following paths are intentionally excluded from Git:

- `data/sec/`: exact SEC response archives and bulk downloads;
- `data/raw/`, `data/interim/`, and `data/processed/`: local research builds;
- `data/market/`: market-data-provider responses and derived local stores;
- `output/`, `outputs/`, and `tmp/`: generated reports and temporary renders.

SEC material can be collected again from the official EDGAR interfaces by an
identified client that follows SEC fair-access requirements. Market data must
be obtained with the user's own provider credentials and used under that
provider's applicable terms. Do not upload provider response payloads, API
keys, or artifacts whose redistribution rights have not been verified.

## Reproducibility boundary

The deterministic simulation, public SEC example slice, research code, tests,
method documents, frozen version identifiers, and aggregate empirical results
are suitable for the public repository. Rebuilding the full empirical event
ledger requires independently collected point-in-time SEC and market data.
