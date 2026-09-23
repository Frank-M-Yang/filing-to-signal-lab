# First-Public Clock v1 — rule-based earliest SEC disclosure

## English

### Why this audit exists

The financial panel times every quarterly observation at its 10-Q/10-K SEC
acceptance. Many issuers disclose the same period's results earlier, in an
earnings release furnished on Form 8-K under Item 2.02. Earlier rounds treated
this as a per-event review (30 events, publisher pages, one rebuilt event).
This audit asks whether the gap is an edge case or structural, by applying one
explicit rule to the whole local SEC catalog.

### Rule

For every **original** (non-amended) 10-Q/10-K:

- Candidate releases are original 8-Ks whose item list contains Item 2.02.
- A candidate must be accepted strictly after the issuer's previous original
  periodic report, on or after the New York day following the fiscal period end,
  at most 90 days before the report, and no later than the report.
- The earliest qualifying candidate is the release clock. The number of
  qualifying candidates is kept as an ambiguity measure.
- `earliest_disclosure_at` is the release clock when one exists, otherwise the
  report's own acceptance.

A periodic report is identified by `(cik, accession)`, never accession alone:
5,246 combined filings list one accession under several co-registrant CIKs, and
only an issuer's own releases may move its clock. Four reports whose catalog
period end is empty are kept with status `periodic_report_missing_period_end`
and their later report clock; they are neither dropped nor guessed.

### Results — whole catalog

Catalog `sec-filing-catalog-v2-e863ecb81e963c7d`, 262,940 original periodic reports:

| Form | Reports | With release | Same NY date | Lead ≥1 day | Lead ≥3 days | Median lead | p90 lead |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10-Q | 191,412 | 56.3% | 45.1% | 45.8% | 33.7% | 0.93 d | 15.1 d |
| 10-K | 71,528 | 51.6% | 25.5% | 70.3% | 62.5% | 8.69 d | 45.0 d |
| All | 262,940 | 55.0% | 40.1% | 52.0% | 41.0% | 1.02 d | 23.1 d |

Shares after "With release" are among reports with a release. For 9.5% of
matched releases the 8-K was furnished after the issuer-reported event date, so
the true announcement preceded even the 8-K acceptance. 6.7% of matched reports
(10-K: 11.9%) have more than one qualifying release; the earliest is used.

### Results — frozen execution cohort

The 316 frozen execution events (64 issuers) were scheduled twice under the same
date-based `ExecutionContract` (second session after the New York announcement
date, 20 sessions including entry day). Only the clock differs.

| Quantity | Events |
|---|---:|
| Report clock found in the catalog table | 316 |
| Cohort filing clock equals catalog acceptance | 316 |
| Earlier Item 2.02 release | 269 |
| Release lead ≥ 3 days | 68 |
| **Entry date changes** | **113 (35.8%)** |
| Entry later by ≥ 2 sessions | 71 |
| Entry later by ≥ 10 sessions | 33 |
| Entry later by ≥ 20 sessions | 8 |

For the last eight events, the filing-clock entry comes after the entire
20-session window measured from the release-clock entry has ended.

### Interpretation

- The gap is structural, not a set of edge cases: about a third of 10-Qs and
  more than half of 10-Ks follow an earlier release by at least three days.
- For affected events, "post-filing" reaction and drift windows start after the
  results were already public. This would bias measured effects toward zero and
  may partly explain the null results; this audit does **not** test that.
- Market features labelled pre-event relative to the 10-Q can contain the
  post-release price reaction for these events. They must be rebuilt against
  the earliest disclosure clock before any new return experiment.
- About 45% of periodic reports have no qualifying release; for those issuers
  the 10-Q/10-K is plausibly the first disclosure and the original clock stands.

### Boundaries

- Both clocks are operational proxies. Item 2.02 permits furnishing up to four
  business days after a public announcement, and an earnings release may reach
  newswires before EDGAR acceptance. `first_public_certified` stays false.
- The rule times disclosure; it does not verify that a release contains the
  same values as the periodic report (some releases omit gross margin). The
  earlier cell review matched 26 of 27 checked values.
- A qualifying candidate can be a preliminary-results or other Item 2.02
  release; the multi-candidate share is reported rather than hidden.
- No prices or returns are loaded. Admission gates stay closed. Previously
  observed outcomes remain training-only; re-analysing them with this clock is
  diagnostic, not validation.

### Reproduce

```powershell
uv run python -m scripts.audit_first_public_clock --catalog-root data/processed/sec_universe/sec-filing-catalog-v2-e863ecb81e963c7d
uv run python -m scripts.compare_first_public_execution --clock-root data/processed/first_public_clock/<bundle>
uv run pytest tests/test_first_public_clock.py tests/test_first_public_execution.py
```

Outputs are content-addressed bundles under `data/processed/first_public_clock/`
and `data/processed/first_public_execution/`. The tests use synthetic catalogs
and calendars and need no local data.

### Next decision

Adopt the earliest disclosure clock as a versioned alternative event clock in
the clock-layer mechanism, rebuild features and market cutoffs against it, and
define admission exit criteria as rule plus sampled value agreement plus
sensitivity bounds rather than per-event certification.

## 中文

### 为什么要做

主面板把每个季度观测的"信息时间"定为 10-Q/10-K 的 SEC 接受时间，但很多公司会更早通过
8-K（Item 2.02）发布财报。此前的做法是逐事件复核（30 个事件、发布方网页、重建 1 个事件）。
这一轮用**一条明确规则**覆盖整个本地 SEC 目录，判断这是个例还是结构性问题。

### 规则

对每份**原始**（非修正）10-Q/10-K：候选发布必须是含 Item 2.02 的原始 8-K；接受时间须晚于
该公司上一份原始定期报告、不早于财政期末次日（纽约日期）、不早于报告前 90 天、不晚于报告本身；
取最早的合格候选，并保留候选数量作为歧义度量。定期报告以 `(cik, accession)` 识别，因为 5,246 份
合并申报在多个共同申报人名下共用同一 accession。4 份目录期末为空的报告保留，标记状态并沿用
较晚的报告时钟，不删除也不猜测。

### 全库结果

26.3 万份原始定期报告中：**约三分之一的 10-Q、六成以上的 10-K，在更早至少 3 天已有财报 8-K**
（10-Q：33.7%，10-K：62.5%）。另有 9.5% 的 8-K 晚于发行人自报的事件日才提交，说明真实公告
比 8-K 接受时间还早。

### 冻结执行队列结果

316 个冻结执行事件（64 家发行人）在同一份日期制执行契约下各排一次日程，只有时钟不同：
**113 个（35.8%）入场日改变**，71 个晚了至少 2 个交易日，33 个晚了至少 10 个交易日，
8 个晚了至少 20 个交易日——按旧时钟入场时，从真实公告算起的整个 20 日持有窗口已经结束。

### 含义与边界

- 这是结构性偏差，不是个案。受影响事件的"公告后反应/漂移"窗口开始于信息公开之后，
  会把测得的效应压向零，**可能**是此前无效结论的部分原因，但本审计没有检验这一点。
- 相对 10-Q 标记为"事件前"的市场特征，可能已包含财报发布后的价格反应，新的收益实验前必须
  按首次披露时钟重建。
- 约 45% 的定期报告没有合格的财报 8-K，这些公司的 10-Q/10-K 很可能就是首次披露。
- 两种时钟都只是代理；8-K 最多可在公告后 4 个工作日提交，`first_public_certified` 保持 false。
  规则只确定时间，不核对新闻稿数值是否与定期报告一致。
- 未加载价格或收益，准入门保持关闭；已观察的结果仍只能训练，用新时钟重分析只算诊断。

### 下一步

把首次披露时钟作为版本化的备选事件时钟接入时间层，据此重建特征与市场截止时间；并把准入门的
退出标准定义为"规则 + 抽样数值一致性 + 敏感性区间"，而不是逐事件认证。
