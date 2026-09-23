# Execution contract migration and release-clock evidence v1

## English

This is an outcome-blind **schedule migration**, not a new return study. It keeps
all **316 previously frozen execution events from 64 issuers**, including events
without usable downstream returns. It does not reselect the 227-event market-state
subset, overwrite the original execution plan, or change forecasts or trading policy.

### Explicit contract

`execution-contract-v1-after-local-date` means:

- Convert the supplied announcement timestamp to `America/New_York`.
- Exclude the entire local announcement date, including pre-open announcements.
- Enter at the unadjusted open of the **second exchange session strictly after that date**.
- Count entry as holding session 1; exit at the unadjusted close of holding session 20.
- Use the pinned venue calendar, including holidays, DST and early closes. Missing
  clocks or incomplete schedules are not filled in.

For example, a Friday pre-open disclosure leads to Tuesday entry in an ordinary
week. The legacy second-future-open rule counted Friday's open and entered Monday.
The two definitions are kept separate; the new contract is not silently installed
in legacy ledgers. These are scheduling conventions, not claims that one maximizes returns.

### Observed comparison

|Check|Result|
|---|---:|
|Frozen execution events retained / legacy windows reproduced|316 / 316|
|Issuers|64|
|Windows changed by the date-rule migration at the same proxy clock|70 (22.2%)|
|Target filing clocks covered by the prior source-clock review|2|
|Target clock timestamps corrected in the isolated view|2|
|Additional window changes caused by those corrections under the new rule|0|
|New windows differing from the stored legacy windows|70|
|Publication-review events retained|30|
|Risk-prioritized publisher pages requested / successful raw snapshots|3 / 1|
|Certified first-public timestamps / events admitted to new returns|0 / 0|

The two reviewed target clocks are OCUL and X. The other **314 target clocks remain
unreviewed catalog proxies**; candidate 8-K reviews must not be counted as reviews
of their target 10-Qs. A resolved schedule is not a certified announcement clock.
All 316 original entry/exit windows reproduce before comparison. The 70 changed
windows have a separate evidence-revalidation queue; existing identity, prices,
actions, delisting and benchmark evidence is not automatically approved for a
different interval. All events remain barred from new returns in this migration.

### Public-time evidence: a bounded result, not a completed audit

The three previously flagged timing-risk cases were selected without looking at
returns. Every original publication-review event remains in the 30-event queue.

- **USNA:** the [issuer's Q1 2025 release](https://ir.usana.com/news-events/press-releases/detail/781/usana-health-sciences-reports-first-quarter-2025-results)
  was saved with its exact response body and SHA-256. The reviewed quarter, units,
  revenue and gross-profit cells agree with the frozen SEC comparison. The HTML
  time attribute says `2025-04-22 16:05:00` **without a timezone**. It is retained
  literally; neither UTC nor Eastern Time is guessed. No public-time bound is asserted.
- **MBUU and UFPT:** their frozen newswire requests timed out. The failure receipts
  are preserved. Web-search visibility does not substitute for a hash-verified
  local document; no publisher clock is admitted for either event.

The USNA page was captured in September 2026, not contemporaneously in April 2025.
A current publisher timestamp, even if later resolved to a timezone, would still
not by itself prove the first public disclosure across all channels. Conference-call
times and scheduled publication times must not replace observed release times.

### Reproduction and artifacts

No API key, SEC contact email or network request is needed for the offline build:

```text
uv run python -m scripts.build_sec_execution_migration
```

The empirical build needs the local immutable bundles pinned in
`sec_execution_migration_v1.json`. They are deliberately not committed. Source
manifests, artifacts, source requests and reviewed HTML hashes are verified before
use; implementation hashes and all event IDs are frozen in a separate plan.

For a public checkout without those private/local datasets, the synthetic tests
exercise the full pipeline and do not require downloading market data:

```text
uv run pytest tests/test_execution_migration.py
```

Outputs under `data/processed/sec_execution_migration/` include:

- `migration_ledger.json`: all original and proposed windows, separated rule and
  clock effects, and explicit closed admission flags.
- `changed_window_evidence_requests.json`: provisional intervals requiring fresh
  evidence validation; not an authorization to collect prices or trade.
- `release_source_checks.json` and `publication_review_queue.json`: exact page
  reviews, failures and unresolved work, without dropping incomplete events.
- `report.json`, `readout.md`, `output_manifest.json`: counts and provenance.

Optional, explicit recollection of just the three frozen public pages is available:

```text
uv run python -m scripts.collect_release_clock_evidence --collect
```

It creates new immutable response/receipt bundles; it does **not** repoint the
offline migration to newly downloaded content. Review and pin new evidence first.
It does not contact SEC endpoints or use personal contact information.

### Meaning and next gate

The positive result is a reproducible identification of a material execution-definition
difference, before it can be mistaken for a model improvement. It is **not evidence
of alpha**, nor proof of higher or lower returns. Old performance outputs are left intact
and are not represented as results of the new convention.

Next: retain missing-source failures, obtain explicit publisher-timezone evidence,
check earlier disclosure channels and accounting scope, and reconstruct forecasts
using strictly prior information. Only after the information-set checks and the
relevant market-window checks pass should a separately versioned return ledger be
built on a frozen common sample. Do not tune models against the migration comparison.

## 中文

本轮是**交易日程口径迁移**，不是新一轮收益实验。我们保留原先冻结的64家发行人、316个执行
事件，不只挑选已具备市场特征或收益的227个子样本，也不覆盖原数据、预测、收益和交易政策。

新的独立约定是：先把公告时间转换为纽约当地日期，**当天不算，之后第二个交易日开盘入场**；
入场日计为第1个持有日，第20个交易日收盘退出。使用未复权开盘/收盘价格类型，分红、拆股、
退市仍须另外处理。普通一周中，周五盘前公告按新规则周二入场；旧“第二个未来开盘”算法会
把周五开盘也算进去，因此周一入场。两者没有偷偷替换，而是明确分版本对照。

结果：316个旧窗口全部复现、事件全部保留；其中**70个（22.2%）窗口会因规则改变而变化**。
此前核验的两条目标财报时钟修正，在这个新日期规则下没有额外改变交易日。
但只有这两条目标财报接收时钟已在前一轮逐源核验，其他314条仍是未审代理，不能称为已认证公告时间。

公开时间核验仍保留30个事件。本轮优先抓取USNA、MBUU、UFPT三个已有时序风险案例：

- USNA原网页已保存，收入和毛利润与此前SEC原始数值对得上；但HTML时间字段没有时区，不能猜。
- MBUU和UFPT本地下载超时，失败记录保留；检索能看到新闻稿，不代表本地已获得可复核的原文证据。
- 当前网页不是当年的抓取记录，发布标签也不自动等于全市场首次公开时间。

因此本轮**没有新增获准回测的事件，没有重算收益，也没有新的Alpha结论**。价值在于提前识别
会影响22.2%事件窗口的口径差异，避免把换了一套交易日期误当作预测模型变好了。

下一道关口仍是：补齐原文与明确时区、检查更早披露及会计口径、按正确的信息可用时间重建预测，
再核验新窗口的证券身份、价格、公司行动、退市与基准。通过后另建新版收益账本，不沿用旧收益
声称新版有效。本轮不需要用户提供密钥；未提交或推送GitHub。
