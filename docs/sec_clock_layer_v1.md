# Versioned SEC Clock Layer and Fixed-Specification Replay

## English

### Scope

This follows the [source-clock audit](sec_clock_provenance_v1.md). The five
evidence-bound corrections are now applied to an **isolated experimental view**,
not the original catalog or any trading policy. The view retains all 34 reviewed
filings, original literals, raw-source hashes, reconciled proxies, and unresolved
states. There is no global timezone shift and no new first-public timestamp.

We reproduced the existing 3,254-event quarterly history before changing any clock.
Then we replayed the identical original accounting observations with the identical
company-history OLS specification and hyperparameters, rebuilding:

- Same-quarter prior-year joins and Level/Delta.
- Strictly earlier company-history features and their cutoff timestamps.
- Expanding out-of-sample forecasts, training endpoints and generation proxies.
- Sigma from strictly earlier out-of-sample errors and both Innovation representations.

The 30-event disclosure review sample remains fixed; the 3,254 rows are the existing
history needed to reconstruct pooled forecasts, **not 3,254 newly audited filings**.
No market-return values were loaded. No model was selected by profitability.

### Findings

Baseline reproduction passed for every compared field, using absolute numeric
tolerance `1e-10`, relative tolerance zero, and identical missingness/event IDs.

| Quantity | Rows changed by the clock overlay |
|---|---:|
| Filing/actual acceptance proxy | 2 |
| Q4 prior-YTD source timestamp | 1 |
| Feature cutoff | 1 |
| Training endpoint | 2 |
| Reconstructed forecast-generation time | 2 |
| Prior OOS error-history endpoint | 2 |
| Level / Delta / forecast / sigma / raw and standardized Innovation values | 0 above tolerance |

The corrected history still contains 2,449 forecasts and 2,399 standardized
innovations. The reconstructed timestamp inequalities pass. This is not a test
that all inputs were public at those times: most history clocks are unreviewed
acceptance proxies, and earlier earnings releases can precede the target 10-Q.

In this bounded perturbation, financial values and model outputs did not materially
change. Clock changes still propagate to future features and other issuers' pooled
training metadata. A synthetic crossing-event test confirms that training counts,
predictions and sigma do change when a corrected clock crosses another event;
the replay is not merely changing displayed labels.

### Execution-convention issue surfaced

We compare two rules explicitly rather than silently replacing one with the other:

1. **Legacy implementation:** second regular open strictly after the timestamp.
   For pre-open news, today's open is counted as the first future open.
2. **Date-based conservative contract:** second trading session strictly after
   the announcement's New York calendar date. Today's session is never counted.

For X's May 2, 2025 10-Q, the corrected proxy moves from 07:56 to 11:56 New York
time. Under the legacy rule, entry moves from May 5 to May 6; under the date-based
rule it is May 6 in both cases. Both hold for 20 sessions, including entry day,
with unadjusted-open entry and unadjusted-close exit price types. These are schedule
requirements, not observed prices or a claim that these are already valid trades.

For X's May 1 earnings 8-K, classification changes from intraday to after close,
and the previous-close/next-close bracket shifts. A daily-close bracket is not an
isolated event reaction: it can include unrelated/pre-announcement trading. Across
the five corrections, two session classifications, one legacy entry date and one
daily-close bracket change. No date-based entry dates change.

The historical target-event exchange mapping is reused for this clock-sensitivity
calculation. Earlier candidate 8-Ks are not independently certified security mappings,
and non-earnings documents are not promoted to earnings events.

### Boundaries and safety

- Original catalog, source snapshots, features, forecasts, return ledgers and policies
  are unchanged. New content-addressed bundles preserve input and implementation hashes.
- The synthetic tests need no credentials or private datasets. Source corruption,
  ambiguous identity, unresolved required clocks, changed sample membership, failure
  to reproduce baseline, or a no-longer-prior Q4 source stops the replay.
- Replayed forecast timestamps are reconstructed historical generation proxies,
  **not contemporaneously captured model-run logs**.
- Financial values and fact selection are held fixed to isolate clock effects;
  this is not a new full fact-selection/financial-statement review. The original
  company-history specification is retained, with no new industry features.
- Existing market-feature cutoffs are checked against replayed filing proxies,
  but market features and returns are not rebuilt. Nothing here revalidates them
  against an earlier public earnings release.
- `model_admission`, `trading_admission` and `public_time_certified` remain false.
  The output panel is deliberately named `disarmed_replay_panel.parquet`.

### Reproduce

With the preserved inputs pinned in [configuration](sec_clock_layer_v1.json):

```powershell
uv run python -m scripts.build_sec_clock_layer
```

For a public, synthetic test without any downloaded dataset:

```powershell
uv run pytest tests/test_clock_layer.py
```

Outputs under `data/processed/sec_clock_layer/` include the frozen plan, clock view,
field-level replay differences, fixed-sample chronology, window sensitivity,
disarmed replay panel, report, and Chinese readout. Repeating with identical inputs
and code reuses the same bundle; an altered existing bundle is rejected.

### Next decision

Make the entry convention explicit in a versioned execution contract before
rebuilding any return ledger. Continue accounting-scope and earliest-public-release
reviews; only then decide whether a revised performance experiment is admissible.
This round is a bounded robustness/control result, **not new evidence of alpha**.

## 中文

这一轮不再只是提出修正，而是建立了一个**独立的实验时间层**，并完成下游重放。
原始主库、旧预测、收益账本和交易规则都没有被覆盖。

先成功复现原来的3,254个历史事件，再只修正已有证据支持的5条时间记录，重新计算
同比滞后、公司历史特征、逐事件样本外预测，以及只使用过去样本外误差的标准化。
这3,254个事件是原有训练历史，不是新增或全部完成原文审查的财报。

结果分成两部分：

- **模型数值没有实质改变**：Level、Delta、预测值、sigma、原始与标准化Innovation均无
  超过绝对容差1e-10的变化。不能把这说成模型变得更赚钱。
- **时间链条确实改变**：2个财报事件的代理时钟、1条特征截止、2条训练/预测生成时间、
  2条历史误差截止等需要更新。影响不只停留在X和OCUL本身，还能传到其他公司的合并训练。

另外查出一个执行约定的差异：旧程序把“盘前公告当天的开盘”计作第一个未来开盘；
而“公告日期之后第二个交易日”不计当天。X的一份10-Q按旧算法会从5月5日入场变为
5月6日；按后一种较保守约定，修正前后都是5月6日。现在把两种约定分别展示，未暗改规则。

这意味着：**我们有了可验证的局部稳定性结果，也找到了具体的执行时间风险。**
但SEC接收时间仍不等于市场首次知道信息的时刻；重放预测也不是当年真实留存的模型日志。
没有重新计算收益，没有发现新Alpha，更没有批准实盘使用。

下一步应先把入场日期定义写成统一、版本化的规则，并继续补首次公开时间和财务口径证据，
再决定是否重建收益账本，不能直接沿用旧收益来评价新时间层。
