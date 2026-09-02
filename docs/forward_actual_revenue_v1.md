# Forward accession-level revenue actuals v1

This layer converts a future filing candidate into a revenue actual without
relaxing the point-in-time clock.

## Admission contract

- The consensus snapshot must predate the SEC filing `accepted_at` timestamp.
- Only the first original `10-Q` or `10-K` is eligible for the unbiased actual.
  Amendments remain evidence for a later look-ahead/revision analysis.
- The companyfacts response bytes and SHA-256 are preserved before parsing.
- A fact must carry the exact filing accession, `us-gaap` taxonomy, USD unit,
  SEC report-period end, and a supported standardized revenue tag.
- A direct quarterly fact must span 70–110 days. A 10-K annual value is never
  relabelled as Q4 revenue. Q4 is admitted only as annual revenue minus a
  same-tag, same-unit 240–300 day fact from the strictly earlier original 10-Q;
  the prior report date must be 60–130 days before fiscal year-end. Otherwise
  the annual value remains explicitly pending.
- SEC `filed` is source metadata, not the availability clock. Availability is
  the accession's SEC acceptance timestamp from the monitored submissions file.
- This layer loads neither prices nor returns and cannot establish alpha.

## Run

Build the current no-network readiness state:

```powershell
uv run python -m scripts.build_forward_actual_revenue
```

After the filing monitor detects a future candidate, collect companyfacts using
an identified SEC User-Agent:

```powershell
$env:SEC_USER_AGENT = "Filing-to-Signal Lab your-real-email@example.com"
uv run python -m scripts.build_forward_actual_revenue --collect
Remove-Item Env:SEC_USER_AGENT
```

The command is resumable. Distinct source bytes are stored under content hashes;
existing immutable snapshots are never overwritten.
