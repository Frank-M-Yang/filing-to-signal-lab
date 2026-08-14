# Revision Tail Source Audit v1

## Why this audit exists

The first walk-forward study found 15 resolved events whose later gross-margin
change exceeded the event-time 95% buffer.  A Companyfacts value change is not, by
itself, evidence of an economic correction.  This audit reads the original and
later primary SEC filings before deciding what changed.

Every judgment is tied to both accessions, the SEC document URLs, and SHA-256
hashes of the exact bytes preserved under `data/sec/raw_filings`.  The complete
event-level evidence is in `reviewed_tail_events.csv` in the v2 output directory.

## Result

| Source-audited family | Events | Meaning |
|---|---:|---|
| XBRL semantic collision | 4 | A disaggregated revenue subtotal was mistaken for consolidated revenue. The audited revision is reset to zero. |
| Discontinued-operations scope recast | 6 | A later filing redraws prior continuing operations after a separation or business sale. |
| Presentation reclassification | 5 | Revenue or cost moves between statement lines while the underlying total economics remain unchanged. |
| Verified economic error correction | 0 | None of the reviewed tail events establishes a correction of the underlying economic fact. |

The four semantic collisions are ADM FY2023 Q3 and FY2024 Q2, plus CHSCP FY2025
Q2 and Q3.  The later filings retain the original consolidated totals; the
Companyfacts selection had picked ASC 606 revenue-disaggregation subtotals.

The six scope recasts are three JNJ/Kenvue events, two PAGP Canadian NGL events,
and one HON/Solstice event.  The five presentation events are HON's separate R&D
line, two Chevron cost/tax reclassifications, and two Pfizer royalty-revenue
reclassifications.

## Effect on the safety-margin result

After the four extraction artifacts are corrected:

- the 95% median event-time buffer remains **0.166 gross-margin percentage points**;
- empirical coverage rises from **96.4% to 97.4%**;
- resolved direction flips fall from **4 to 3**;
- robust-classified direction flips fall from **4 to 3**;
- changed resolved outcomes fall from **49 to 45**.

The main 95% buffer therefore was not created by the four largest-looking data
errors.  Its interpretation does change: the remaining reviewed tail measures
the risk that future accounting scope or presentation redraws the historical
signal, not a measured rate of financial misstatement.

## Candidate second-stage gate

We tested one deliberately narrow candidate using only information already public
in the original filing: abstain when the filing explicitly discloses a planned
separation or structural change.

- It flags 4 of the 11 valid published-history redraws (36.4% recall).
- It flags 0 of the 3 post-audit direction flips.
- A broad non-tail filing-text control sample is not yet available, so its false-
  positive rate cannot be estimated.

The candidate therefore has status
`do_not_deploy_needs_broad_event_time_text_controls`.  It is not applied to
decisions.  The first-stage walk-forward numeric buffer remains the only active
decision diagnostic.

## Evidence boundary

This is a financial-data stability study.  It contains no point-in-time prices,
returns, historical constituents, delisting returns, transaction costs, or
portfolio claims.  It does not identify fraud and must not be read as investment
advice.
