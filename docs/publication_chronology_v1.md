# Publisher clocks versus filing clocks: evidence follow-up

## English

This outcome-blind follow-up keeps the same 30-event publication-review cohort and
its 316-event upstream execution cohort. Three existing timing-risk cases were
prioritized; no event was replaced based on returns or download success.

### Result

We now have four saved publisher pages across USNA, MBUU and UFPT, including three
new successful downloads. Six exact revenue/gross-profit cells, two per company,
match the pinned original SEC fact review. Seven source-request receipts are
included across the selected batches; three failed wire requests remain visible.
The initial sandbox connectivity failures are also retained in the raw directory,
but are not counted as publisher responses in this seven-receipt evidence set.

|Case|Verified source evidence|Chronology finding|
|---|---|---|
|USNA|Full financial table; datetime has no timezone|Reconstructed forecast is later even across the declared ±14-hour offset sensitivity range|
|MBUU|Full financial table; only a publication date|Same-day ordering remains unresolved; conference-call time is not used|
|UFPT issuer site|Explicit-offset CMS publication field; rounded financial summary|Site publication time must not be attached to exact cells obtained elsewhere|
|UFPT issuer-linked newswire copy|Full table and explicit EST publication label, to the minute|Reconstructed forecast is more than three days after this publisher label|

The sources are the [USNA issuer release](https://ir.usana.com/news-events/press-releases/detail/781/usana-health-sciences-reports-first-quarter-2025-results),
[MBUU issuer release](https://malibuboatsinc.com/investor-information/earnings-news/news-details/2024/Malibu-Boats-Inc.-Announces-First-Quarter-Fiscal-2025-Results/default.aspx),
[UFPT issuer summary](https://ufpt.com/news/ufp-technologies-announces-strong-q3-2025-results/)
and its [linked full newswire release on Nasdaq](https://www.nasdaq.com/press-release/ufp-technologies-announces-strong-q3-2025-results-2025-11-03).

### UFPT: a concrete, reviewable discrepancy

The issuer-linked full release reports **November 3, 2025, 16:08 EST** (21:08 UTC).
Minute precision is retained as `[21:08:00, 21:09:00)` rather than treating the
displayed minute as an exact second. The original-quarter revenue and gross profit
are USD 154,558,000 and USD 42,747,000. Both match the saved SEC comparison.
The separate issuer page's `datePublished` is **21:35:57 UTC**, 27 minutes 57 seconds
after the start of the newswire label's minute. Its `dateModified` is a third field,
not a substitute publication timestamp. These are different channel labels, not
evidence that the earliest label is the first public disclosure everywhere.

The fixed-spec historical replay places forecast generation, training end and
prior-error-history end at **November 6, 23:29:08 UTC**. That is over three days
after the full-release label, although before the target 10-Q's acceptance proxy.
The feature cutoff is earlier than both. This demonstrates why checking only the
feature cutoff or only the 10-Q acceptance time is insufficient. These are
reconstructed research timestamps, not logs of forecasts actually made in 2025.

Keeping the same date-based execution contract gives a second important difference:

|Clock used for the schedule-only scenario|Second-session entry open (UTC)|
|---|---|
|Target 10-Q acceptance proxy|2025-11-11 14:30|
|Full newswire release's reported label|2025-11-05 14:30|

The proposed release-label scenario exits on December 3 at 21:00 UTC, holding 20
exchange sessions inclusive of entry. These are **provisional schedules**, not
price-ready trades. The filing acceptance field must remain unchanged: disclosure
availability and filing acceptance are distinct clocks.

### What is and is not established

The positive result is stronger source evidence and a concrete diagnostic replay
candidate. A filing-time residual may not represent an innovation at the earlier
earnings release. This is a measurement/chronology issue, not improved alpha.

USNA's timezone is still unknown. Its naive label is tested over a declared
UTC-offset range of `[-14h, +14h]`; MBUU's date-only label additionally covers the
entire stated day. These are **sensitivity scenarios, not certified evidence bounds**.
Neither an inferred timezone nor a conference-call time is written into the data.

All pages were captured in September 2026. Current publisher labels cannot by
themselves certify the historical first-public timestamp, historical page content,
or absence of earlier disclosures. No model, forecast, sigma, return or policy was
changed in this round. First-public certifications and new return admissions remain
zero; the other 27 review events remain in the queue.

### Reproduction

```text
uv run python -m scripts.build_publication_chronology
uv run pytest tests/test_publication_chronology.py
```

The empirical build is offline and needs the local hash-pinned bundles in
`publication_chronology_v1.json`. Raw pages and research bundles remain ignored by
Git. The synthetic tests work without those empirical files or API keys.
Outputs under `data/processed/publication_chronology/` include:

- `source_reviews.json`: publisher clock selectors, precision and separately verified cells.
- `event_chronology.json`: all 30 original events and strict forecast/training/error-clock comparisons.
- `request_receipts.json`: selected successes and failures, with source/config hashes.
- `disarmed_replay_requests.json`: one UFPT clock-sensitivity candidate, with both
  schedule scenarios, a required whole-information-set rebuild and closed return flags.
- `report.json`, `readout.md`, `output_manifest.json`: counts, limitations and provenance.

The next bounded experiment is the UFPT **disarmed publisher-label replay**:
preserve SEC acceptance, change the experimental actual-availability clock, rebuild
lags/forecast cutoff/training/prior-OOS-error sigma, and compare against the untouched
baseline. Do not merely relabel an existing forecast. Earlier-channel checks and
new market-window evidence are still required before any new return study.

## 中文

本轮补下3份新原文，加上已有USNA网页，共核对4份发布来源。三家公司各有收入和毛利润两个
精确数值与此前SEC原始数据对应，共6个单元格；原先30个审查事件全部保留。

最有用的发现是UFPT：公司官网链接的完整新闻稿标注2025年11月3日16:08 EST，而旧研究流程
重建的预测、训练截止和历史预测误差截止是11月6日23:29 UTC，晚了三天多。虽然预测特征截止
早于新闻稿、预测生成早于10-Q提交，但**这并不足以证明它是财务结果公开之前的预测**。

同样采用“公告当地日期之后第二个交易日开盘”规则，UFPT候选入场日会从11月11日变成11月5日。
因此，不能直接给旧预测换一个时间标签，也不能继续沿用旧交易窗口收益。

USNA的标签没有时区；即使按明确的±14小时UTC偏移范围做压力检查，预测仍晚于全部情景。
MBUU目前只有发布日期，同一天内谁先谁后仍不能确认。该压力区间只是研究假设，不是已验证
的真实公开时间上下界。UFPT官网的精确CMS时间也不能替代完整新闻稿渠道的时间，更不能把
网站修改时间、电话会议时间当公告发布时间。

这轮的正面成果是**证据更完整，并确定了一个可独立复算的具体案例**，不是发现盈利能力。
现存网页是2026年抓取的，尚不能据此认证当年的首次公开时刻；没有重算预测或收益，也没有
放开交易或推送GitHub。

下一步已有明确输入：先用UFPT做禁用交易的发布标签时间重放，保留原SEC接收时间，重建
当时可用信息、样本外预测和历史误差波动，再比较新旧结果。取得更早披露核查和新窗口市场证据
之后，才讨论新版收益实验。
