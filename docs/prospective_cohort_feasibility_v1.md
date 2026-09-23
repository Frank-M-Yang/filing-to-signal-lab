# Prospective cohort feasibility v1 / 新前瞻队列可行性

This is an offline, outcome-free **planning audit**, not a new registration, a
model experiment, or an effectiveness report. It leaves the original 37-event
registry and all quality/admission gates untouched. No credentials are needed.

## Run

```powershell
uv run python -m scripts.build_prospective_cohort_feasibility
```

The command selects the quality report by its recorded generation timestamp,
then resolves readiness, registry, and consensus by exact manifest SHA-256.
It verifies reports and input artifacts before calculating anything. To replay
one historical quality bundle, pass `--quality-root <bundle-directory>`.
Other optional paths are `--processed-root`, `--core-root`,
`--provider-audit-root`, and `--output-root` (see `--help`).

Local data are required; this is not the public synthetic demo command. It
does not download missing data. An absent, changed, or mismatched parent fails
closed rather than triggering collection or silently selecting a replacement.

## Inputs and outputs

Inputs are the frozen event identities, quality metadata, core-company identity
columns from `core_asset_82.csv`, aggregate forward-consensus coverage, and the
existing provider entitlement/field decisions. Historical facts, actual revenue,
innovation values, prices, and returns are not inputs. Actual/return loading
flags and populated outcome columns are rejected in numerical inputs.

The versioned bundle under `data/processed/prospective_cohort_feasibility/`
contains `cohort_feasibility.md`, a JSON report, hash manifest, and Parquet tables:

- Issuer-level quality counts.
- Core-company provider coverage, retaining unknown coverage as unknown.
- A non-registered queue of unprobed, mapping-eligible issuers in the original
  core selection order, not signal/performance order. Missing original ranks
  remain missing; those issuers follow ranked issuers in frozen CSV row order.
- Conditional capacity scenarios and request-budget scenarios.
- Copies of existing provider decisions, explicitly not new entitlement tests.

Source and implementation hashes determine the immutable version. Reuse verifies
the report and output hashes; it never overwrites a distinct bundle. Nothing is
added to the daily production sequence or the frozen registry.

## Method and limits

Let `q_e = clean events / registered events` and
`q_i = issuers with any clean event / registered issuers`. These are descriptive
fractions conditional on the selected companies and observed capture dates.
They are **not** probabilities for the wider market. `q_i` depends on how many
quarters each issuer contributes. Repeated quarters are not independent trials.

For assumed additional event/issuer retention `r` and assumed shares `p+`, `p-`
of positive/negative innovations **among all complete events**, calculate:

```text
required complete = max(30, ceil(8 / p+), ceil(8 / p-))
raw events       = ceil(required complete / (q_e * r))
raw issuers      = ceil(15 / (q_i * r))
```

Zero quality fractions give an unknown/unbounded requirement, not division by
zero. Both retention and sign-share grids are explicit sensitivity assumptions.
Using the same additional `r` for events and issuers is illustrative, not an
empirical link between their losses. No actual/innovation values are inspected
to estimate tail frequencies. Counts exceeding these expectation thresholds do
not guarantee realized gate passage, provide statistical power, or establish
alpha. Downstream actual matching, staleness, identity/actions/delisting, prices,
and costs may produce further, correlated attrition.

The original core is a selected, large-company/data-completeness sample. Enlarging
within that core does not remove survivorship or representativeness concerns.
Prospective securities that later delist must still be retained and audited.

## Provider and operational evidence

[Alpha Vantage's official support page](https://www.alphavantage.co/support/)
lists a general free allowance of 25 requests/day (checked 2026-09-14).
Its [estimate endpoint documentation](https://www.alphavantage.co/documentation/#earnings-estimates)
describes annual/quarterly revenue and EPS estimates and analyst counts. This is
not evidence of analyst identities, market-first-availability clocks, or
historical revenue snapshot versions. Current local evidence covers only the
previously probed companies; unprobed companies remain unknown.

The audit does not verify the current key's entitlement. Budget 20 describes the
existing software plan; 25 is a general published allowance; 40 is hypothetical.
One request per issuer/day plus one retry reserve gives at most four new daily
probes beside the existing 20 under a 25-request assumption. A single probe is
not admission: repeated captures, exact quarter/metric/currency/unit matching,
and source provenance still need validation. No actual budget or schedule changes
are made. A larger daily universe requires budget approval or a separately
tested, registered cadence that preserves the frozen seven-day staleness rule.

## 下一步

先确认预算与新探测安排，按原名单顺序测试未知公司，保留所有失败与缺字段。
只有在口径、财政季度、来源哈希和重复前瞻观察均可验证后，才单独注册新队列。
目标季度不得凭空推算；新事件必须与原 37 个事件及已观察事件做 ID 交集检查。
正负创新不足不能靠收益导向的替换或降低门槛来解决。本轮不升级任何质量状态。
