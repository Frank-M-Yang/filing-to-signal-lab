# Prospective Consensus Quality v1

## Purpose

This layer distinguishes a numeric analyst-consensus revision from a change in
the data provider's structure or analyst coverage. It uses no actual revenue,
price, return, or model outcome.

All evidence is retained. A non-clean event continues through the observation
pipeline, but it cannot enter Innovation, paper-decision, or trading-candidate
states until the quality issue is resolved by a future documented process.

## Event statuses

- `clean`: no structural rule fired; downstream admission is allowed.
- `review_required`: the change may be real but its interpretation depends on a
  changed analyst set or another non-terminal structural warning.
- `quarantined`: provider row identity, large analyst-set change, coupled range
  shock, horizon change, or stale pre-acceptance evidence makes the observation
  unsafe for downstream admission.
- `insufficient_history`: fewer than two independent capture dates exist.

Only `clean` is admitted downstream in v1.

## Frozen rules

| Rule | Severity | Frozen threshold |
|---|---|---|
| Provider horizon changed | Quarantine | Any change |
| Source occurrence changed | Quarantine | Any change |
| Large analyst coverage change | Quarantine | At least 25% |
| Range shock with coverage change | Quarantine | Range ratio outside 0.50–2.00 plus count change |
| Analyst coverage changed | Review | Any smaller count change |
| Consensus range-width shock | Review | Range ratio outside 0.50–2.00 |
| Large consensus level change | Review | At least 10% absolute revision |
| Stale pre-acceptance consensus | Quarantine | More than 7 calendar days |

These thresholds are explicit calibration assumptions. They were not selected
from return outcomes and cannot be silently relaxed.

## Current result

Across the 37 frozen v3 events after three independent capture dates:

- 33 are `clean`;
- 2 MCK periods are `review_required` because analyst coverage changed from 12
  to 14;
- NVDA 2026-10-31 is `quarantined` because analyst coverage fell from 43 to 27,
  the provider source occurrence changed, and the consensus range contracted;
- WMT 2026-10-31 is `quarantined` because the provider source occurrence
  changed.

This is a data-quality result, not evidence of positive or negative returns.

## Outputs

- `prospective_consensus_event_quality.parquet`: one assessment per event;
- `prospective_consensus_change_evidence.parquet`: every consecutive-date comparison;
- `prospective_consensus_quality_rules.parquet`: frozen rule catalog;
- `prospective_consensus_quality_gates.parquet`: audit integrity gates;
- `prospective_consensus_quality_report.json`: summary and immutable lineage.

## Manual build

```powershell
uv run python -m scripts.build_prospective_consensus_quality `
  --readiness-root <prospective-readiness-root>
```

