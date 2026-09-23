# Revenue consensus coverage probe v1 / 收入一致预期覆盖探测

This is a deliberately isolated ingestion experiment. It does not expand the
original 37-event registry or promote quality states. No historical returns,
actuals, or innovations are used to select companies. No model is trained.

## Commands

Offline plan and audit (requires the local feasibility and original 20-name
plan bundles):

```powershell
uv run python -m scripts.build_alpha_vantage_coverage_probe
```

Live capture, using only a process-scoped environment credential:

```powershell
# Set ALPHA_VANTAGE_API_KEY in this process without printing it.
uv run python -m scripts.build_alpha_vantage_coverage_probe --collect --limit 4
```

Default `--offset 0` selects the first four names of the frozen unknown-company
queue, not four successful responses. It never fills failures with other names.
Same-UTC-day reruns skip attempted names and do not retry transport failures.
Running the same command on another UTC day provides a second capture of the
same batch, not a new independent filing event. Use an explicitly approved
`--offset 4` for the next fixed slice; there is no automatic daily scheduling.

The new responses stay in `data/market/alpha_vantage_coverage_probe/`; audits
and a readable `coverage_probe_findings.md` stay under
`data/processed/alpha_vantage_coverage_probe/`. These paths remain Git-ignored.
No response or estimate enters the original forward consensus audit directory.
Ordinary responses have exact bytes and SHA-256 preserved before parsing. A
distinct request has distinct metadata even when bytes repeat. There is one
explicit privacy exception: provider errors that echo a credential are scrubbed
before persistence. Their manifests distinguish the stored redacted bytes from
the original wire hash/size, with `credentials_redacted=true` and
`original_wire_body_retained=false`. They cannot supply research candidate rows.
If an apparently successful response echoes the key, its whole financial payload
is withheld instead of admitting modified facts. SEC response handling is unchanged.

The second capture on 2026-09-15 exposed this issue in an MPC quota response.
Its unsafe local error body was retired by a narrowly targeted privacy repair;
a sanitized replacement, the previous safe manifest, original wire fingerprint
and dated `privacy_redaction.json` remain. Original error bytes are no longer
recoverable from this bundle; it is not a verbatim raw-error archive. No valid
financial response, forecast value or capture timestamp was changed.

`coverage_probe_continuity.parquet` now compares consecutive attempts for each
issuer, keeping the union of their future-quarter rows. It records actual elapsed
hours separately from distinct UTC dates, whole-response hashes separately from
revenue mean/range/count changes, and source row position separately from analyst
membership. The descriptive `spans_24_hours` flag is not a new admission gate.
An identical response is repeatability evidence, not a new information set.
Changed response bytes may concern EPS or another quarter, not the revenue under
study. An unchanged analyst count does not establish unchanged analyst identities.

Missing, zero-coverage, error and duplicate-quarter endpoints stay visible as
non-comparable pairs. The comparison never skips an intervening failed attempt,
fills a missing estimate forward, replaces a quarterly row with an annual row,
or computes a revision through an ambiguous duplicate. A period disappearing
because it has ended is labelled separately from a still-future missing period.
The mean-change fraction is a within-provider aggregate diagnostic, not a
standardized innovation, analyst-level revision, or tradable surprise.

## Budget protection

Both existing production Alpha Vantage commands now use a common
`data/market/alpha_vantage_request_budget/` ledger. The request client takes an
exclusive filesystem lock through HTTP dispatch, enforces at least 12.1 seconds
between tracked requests, and writes a reservation before sending. An uncertain
transport failure still consumes its attempt. Receipts are separate immutable
files; legacy response evidence is reconciled without double-counting receipts.

The new probe additionally protects all unfinished original 20 daily targets
plus one retry request. Under the local 25-request ceiling, at most four probe
requests remain whether the original daily capture has run yet or not. Provider
rate limits/permissions and HTTP 401/403/429 stop the batch. No automatic retry
or plan enlargement is authorized. The original research configuration and
20-name daily plan are unchanged; the shared counter is a transport safety guard.

This is **repository-local** accounting, not a provider account-wide usage API.
Calls outside these guarded commands and transport failures before this ledger
was installed may be unknown. Set `--external-requests-today N` on the probe or
existing consensus command, or `--alpha-external-requests-today N` on the daily
pipeline, to conservatively debit known other use. Keys themselves are never
persisted in this ledger. Do not switch keys to bypass a provider restriction.

A provider refusal overrides a positive local counter. On 2026-09-15 the second
request reported a 25-per-day limit even though the local UTC-day allowance was
not exhausted. A local UTC date boundary is **not** evidence of a provider quota
reset, and no reset time was established. This run stopped at the refusal; it
did not retry or replace the company. The current script does not persist an
account-wide cooldown or automatically resume collection.

For historical manifest compatibility, an `Information` envelope retains the
legacy status `premium_endpoint_or_invalid_key`. The separate `provider_reason`
now identifies an explicit rate-limit message as `rate_limited`; the envelope
alone is not evidence that a key is invalid or a subscription upgrade is needed.

An interrupted process may leave `collector.lock`. Do not automatically delete
it or assume it is stale. Inspect the owning task before manual recovery. A
busy lock fails closed without sending another request.

All day accounting uses the actual UTC clock, not the user's local calendar
date or a requested historical label. Throttling rechecks the date after
midnight. Original collection also stores responses under their actual fetched
UTC day and stops if a response crossed a UTC-day boundary, requiring an
explicit restart. Capturing today never creates a historical snapshot.

## Candidate checks and limits

Only the provider's actual `fiscal quarter` revenue rows with period date at
least the capture UTC date are candidate rows. Positive mean, positive ordered
low/mean/high, and finite positive integer analyst count are basic field checks.
Missing/invalid fields remain rejected rather than imputed to zero. A period
date is not a SEC filing acceptance timestamp and does not authorize admission.

`coverage_probe_field_rejections.parquet` retains per-row reasons, including
provider rows with zero revenue estimate and zero analyst count. Those observed
zeros remain zeros in rejection evidence; they do not mean actual company revenue
is zero or that a valid zero-revenue consensus exists.

The observed response schema has no explicit currency/unit fields, so those
columns stay missing. Currency, unit, revenue accounting scope, fiscal-cycle
alignment and SEC acceptance cutoffs remain **unverified**, even for repeated
candidate captures. Analyst identity is unavailable: stable analyst count or
source row position does not prove a stable panel or genuine within-analyst
revision. `formal_model_admission` and `new_cohort_registered` always remain false.

The first-batch definition review is recorded in
[the calendar and metric review](consensus_probe_definition_review_v1.md).
It identifies unresolved mappings; it does not mark provider fields as verified.

The [provider-definition audit](consensus_provider_definition_audit_v1.md) extends
this check to the original and probe response archives. It separates observed
field presence from verified semantics and includes an unsent support request.

Next: repeat the fixed first batch in real time, evaluate the elapsed interval,
field continuity and matching definitions, then design a separate new-event
registration with original/observed event-ID overlap checks. Do not weaken the
seven-day staleness rule, original gates, or unknown-coverage status to obtain a
larger-looking sample. This is feasibility evidence, not an alpha report.
