# Forward Level / Revision Delta / Innovation ledger v1

This ledger is the first fully forward-only feature table for the analyst-
consensus experiment. It still contains no prices or returns.

For each future SEC event:

- **Level** is the last captured average revenue estimate strictly before the SEC
  acceptance timestamp.
- **Revision Delta** is the change from the latest estimate on an earlier capture
  date to that pre-event Level. Same-day duplicate observations cannot create a
  revision.
- **Raw Innovation** is reported quarterly revenue minus the pre-event Level;
  the percentage form divides by that Level.
- **Standardized Innovation** divides the percentage error by an expanding error
  sigma. The sigma uses only errors from SEC events accepted strictly earlier.
  Same-time events, future events, and in-sample residuals are excluded.

The persisted `forward_innovation_complete_cases.parquet` contains only events
where Level, Revision Delta, and Raw Innovation are all available on the same
Company × Filing Event × Information Set. Return research remains blocked until
the later market-data layer is explicitly built.

Run:

```powershell
uv run python -m scripts.build_forward_innovation_ledger
```
