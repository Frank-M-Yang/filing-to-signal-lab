# SEC Disclosure Follow-up: Document Types, Clocks and Accounting Scope

## English

### Follow-up status

The [clock-provenance audit](sec_clock_provenance_v1.md) has now traced the three
conflicts and both related target 10-Qs to raw source records. It proposes five
acceptance-clock corrections without applying them to the main catalog. The
original candidate sets are unchanged; publication-time and accounting gates remain closed.

### What this round does

The frozen 30-event sample is unchanged. This offline follow-up checks all 32
cached candidate submissions' raw headers and reviews the 12 submissions acquired
in the preceding collection batch. It also rechecks the earlier three table-review
cases. Nothing is selected by returns; missing or difficult events are not replaced.
This is a measurement audit, not a new forecast or return experiment.

The [review specification](sec_disclosure_followup_v1.json) pins the original plan,
field-audit manifest, prior review and exact submission/exhibit hashes. Each table
cell has period, unit and numeric anchors. These bind reviewer context assertions
to evidence; they are not an automatic semantic classifier or an independent
second-person accounting review. Price/return values and fitted forecast values
are not loaded. Only forecast clock metadata is projected for timing checks.

### Important correction to the initial audit

Our earlier wording described SEC acceptance as a proven public-availability upper
bound. That was too strong. Acceptance and public dissemination are distinct, and
the SEC does not guarantee the dissemination delay. Acceptance is retained only
as an operational proxy; no arbitrary fixed delay is added to manufacture a
certified clock. See the [SEC timestamp and dissemination explanation](https://www.sec.gov/about/webmaster-frequently-asked-questions).

Accordingly, the prior USNA/MBUU after-bound findings become **timing-risk flags**,
not proof that the forecasts followed public dissemination. The old bundles remain
unchanged as audit history. Current documentation and newly generated initial-audit
outputs use corrected proxy terminology. All public-availability and first-public
timestamps in this follow-up remain null; model admission remains false.

### Findings

| Check | Result | Meaning |
|---|---:|---|
| Candidate raw headers | 32 checked; 29 consistent, 3 conflicts | Preserve both clocks and quarantine conflicts |
| New document dispositions | 8 earnings, 1 preliminary, 3 other | An 8-K candidate is not necessarily an earnings event |
| Financial table cells, including prior cases | 27 checked; 26 exact original matches | Matching numbers alone do not establish compatible scope or timing |
| Events with an exact value match | 11/30 | Partial coverage, not a training or representative sample |
| Gross-margin forecast proxy checks | 6 checked; 3 timing-risk flags | No first-public forecast certification |
| New models / returns / trades | 0 | No alpha or performance claim |

The three raw-clock discrepancies are:

| Issuer | Accession | Frozen catalog UTC | Raw header interpreted as America/New_York, in UTC |
|---|---|---|---|
| X | 0001104659-25-032901 | 2025-04-08 09:05:05 | 2025-04-08 13:05:05 |
| X | 0001163302-25-000028 | 2025-05-01 16:21:59 | 2025-05-01 20:21:59 |
| OCUL | 0001104659-24-118284 | 2024-11-14 07:10:12 | 2024-11-14 12:10:12 |

The discrepancies have the size of Eastern/UTC offsets. This is a reason to audit
the upstream source and normalization, not evidence that every catalog timestamp
needs shifting. America/New_York handles seasonal offsets and rejects ambiguous or
nonexistent wall times. The original catalog and target 10-Q timestamps have not
been rewritten; target-header reconciliation and candidate-boundary rechecking
remain necessary. Existing catalog clocks cannot be assumed validated by this
small review.

Specific accounting and event distinctions:

- SUN's distribution notice, X's acquisition-review memorandum and EAF's listing
  compliance notice are retained in the inventory but do not set an earnings clock.
  A listing notice is not automatically a delisting outcome.
- EA's January 22 preliminary release includes approximate quarterly revenue before
  its February results release. It remains a distinct preliminary-information
  stage, not a certified final actual or a precise first-public timestamp.
- NVIDIA's release directly reports quarterly operating cash flow of **USD 14.489
  billion**, while subtraction of the preserved current and prior YTD originals
  gives **USD 14.488 billion**. The USD 1 million gap is small, but its cause has not
  been established. No rounding plug, restatement assumption or silent replacement
  is allowed. Absence of a direct companyfacts quarter does not imply no quarter
  was disclosed in the release.
- TT reports USD 850.8 million consolidated earnings versus USD 847.6 million
  attributable to the parent. Its cash-flow headline is continuing-operations YTD.
  Neither scope nor period can be mixed with a parent quarterly income figure by
  assumption. SUN's distributable cash flow is also not GAAP operating cash flow.
- For TTC, NVDA and AMWD, all three forecast clocks precede the matched gross-margin
  acceptance proxy. USNA, MBUU and UFPT have one or more clocks at/after the proxy.
  Neither group is certified before/after first public disclosure.

The source accessions, exhibit hashes, short cell anchors, unresolved-scope notes
and document dispositions are reproducible from the local preserved SEC evidence.
No contact identity is embedded in the specification or report.

### Reproduction

With the pinned local artifacts, no credentials or network access are needed:

```powershell
uv run python -m scripts.review_sec_disclosures
```

Synthetic safety tests do not need those research assets:

```powershell
uv run pytest tests/test_disclosure_review.py tests/test_disclosure_audit.py
```

Content-addressed output bundles are written under
`data/processed/sec_disclosure_followup/`: `header_clock_audit.json`,
`document_dispositions.json`, `metric_cell_review.json`, `event_review.json`,
`report.json`, and a Chinese `readout.md`, with output hashes. Repeated runs reuse
identical bundles; prior outputs are not overwritten. Raw research files are not
included in the public repository.

### Next bounded step

1. Trace X/OCUL catalog clocks to their preserved source and original target 10-Q
   headers. Recheck candidate boundaries before creating any versioned correction.
2. Finish the other candidate content/metric reviews and resolve the documented
   YTD/scope/precision issues. Do not discard unresolved companies for convenience.
3. Obtain independent release/dissemination evidence where feasible, or explicitly
   design a filing-time study with uncertainty and a conservative execution rule.
   An IR dateline or conference-call time alone cannot certify publication time.
4. Only after clock and accounting gates are satisfied, preregister a separate
   hypothesis and fresh validation. This historical audit is not a new holdout.

## 中文

本轮把“下载到了”推进到“逐项核对”，没有训练模型，也没有修改原交易规则。

- **12份新附件**已分类：8份业绩公告、1份初步业绩预告、3份其他事项公告。
- **32份候选原文**都做了头部时钟核对，发现X和OCUL涉及的3份文件存在4或5小时差异，
  已在新审计中隔离，未直接平移全库时间。
- 连同此前3个案例，共核对**27个财务数值，26个与原始财报一致**，覆盖11个事件。
  这不是说11个事件已经完全清洗好或可以训练。
- NVIDIA新闻稿直接披露的季度经营现金流，比原有“两个累计值相减”多100万美元。
  金额差异很小，但原因还不能确认，因此保留两套证据，不自动修补。
- EA在正式业绩公告之前已有初步预告，说明信息可能分多次进入市场；
  SUN可分配现金流、TT持续经营现金流也不能直接当作统一口径的季度经营现金流。

**需要诚实更正此前的一句话：SEC接收时间不能证明市场已经在那一秒拿到了文件。**
所以此前“预测晚于公开披露”的判断要收窄为“相对于接收代理时钟存在风险，需要进一步核实”。
本轮6个能做毛利率时钟检查的事件中，3个有此风险；并非新增了3个已证实的前视泄漏案例。

这轮的价值是找到了会制造假信号的具体数据问题，并留下了可复核证据。下一步优先核对X/OCUL
的上游时间来源和原始10-Q，再继续剩余口径审查。没有新增Alpha结论，尚不开放模型或交易准入。
