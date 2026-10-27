# Disclosure-Clock Return Diagnostic v1

## English

### Question

The validation return ledger (227 events, 48 issuers) timed every event at its 10-Q/10-K
acceptance and concluded `effectiveness_not_established`. The
[first-public clock](first_public_clock_v1.md) later showed that most of those results were
public days earlier through an Item 2.02 earnings 8-K. Did the misaligned clock hide an
effect?

### What was done

The same 227 observed events were re-scheduled twice under the frozen date-based execution
contract (entry at the second session after the New York date, exit at the twentieth
close): once from the filing acceptance, once from the earliest SEC-evidenced disclosure.
Issuer and SPY cash-retained returns were recomputed from the same preserved bars and
corporate actions, so the clock is the only difference. The frozen issuer-cluster inference
(10,000 bootstrap draws, seed 20260814) then ran on one complete-case sample under both
clocks.

Faithfulness check: for the 182 events whose date-based filing window equals the frozen
ledger's window, the recomputed market-excess return matches the ledger exactly (maximum
absolute gap 0.0). Eleven events whose disclosure-clock entry precedes the preserved price
window are excluded, leaving a same-sample of **216 events / 48 issuers**, of which 90
have an earlier entry under the disclosure clock (1 to 24 sessions; median 3).

### Result

Spearman IC of each representation with the 20-session market-excess return, same 216
events, issuer-clustered 95% intervals:

| Representation | Filing clock | Disclosure clock | Difference |
|---|---:|---:|---:|
| Level | +0.012 [−0.121, +0.142] | −0.023 [−0.161, +0.111] | −0.036 |
| Delta | −0.025 [−0.142, +0.088] | −0.017 [−0.134, +0.095] | +0.008 |
| Raw Innovation | −0.004 [−0.132, +0.122] | +0.026 [−0.110, +0.155] | +0.030 |
| Standardized Innovation | −0.004 [−0.132, +0.121] | +0.026 [−0.111, +0.155] | +0.030 |

**The clock change does not alter the conclusion.** Every interval spans zero under both
clocks; the largest shift, +0.03 for Innovation, is a quarter of the interval width.

Two subsets, reported because they were computed, not because they were preregistered:

- **Innovation knowable at disclosure (97 events, 36 issuers).** For 130 of the 227
  events, at least one forecast dependency (feature cutoff, training end, or out-of-sample
  error history end) falls at or after the earliest disclosure: the pooled expectation was
  formed with filings that were not yet public at the release. On the 97 events where it
  was knowable, Innovation IC is +0.040 under both clocks (interval [−0.15, +0.25]).
- **Entry moved earlier (90 events, 28 issuers).** Level IC moves from −0.037 to −0.165
  (interval [−0.346, +0.024]); Innovation from −0.114 to −0.087. A subset-specific
  observation on previously observed outcomes, not evidence.

### Reading

- The release-timing correction was necessary for validity, but on this cohort it does
  not change the answer: no representation of gross margin predicts the 20-session excess
  return under either clock.
- The 130/227 count is a validity finding about the original study that is independent
  of returns: its Innovation feature used information that arrived between the release
  and the periodic report. A full re-timed replay of the forecasts under the disclosure
  clock is the next step before any Innovation claim.
- These outcomes were observed before this diagnostic existed. It is a diagnostic of the
  clock; it cannot select a model or count as validation.

### Boundaries

- Gross margin is assumed public at the release (`release_value_assumption`); that was
  checked on a small reviewed sample (26 of 27 cells), not certified per event.
- Both clocks are SEC proxies. Entry under the disclosure clock is at most 24 sessions
  earlier here; the 8-K may itself trail the announcement by up to four business days.
- No cost, sector neutralization, strategy, or P&L is computed.

### Reproduce

```powershell
uv run python -m scripts.build_disclosure_return_diagnostic
uv run pytest tests/test_disclosure_return_diagnostic.py
```

The synthetic tests cover window recomputation, the earlier-release case, the
innovation-knowable flag, the ledger consistency check (which fails closed on a 1e-6
discrepancy), and missing-bar exclusion.

## 中文

### 问题与做法

验证账本的 227 个事件（48 家发行人）此前按 10-Q/10-K 接受时间计时，结论是"有效性未建立"。
[首次公开时钟](first_public_clock_v1.md)表明这些结果多数在几天前就已通过财报 8-K 公开。
时钟错位有没有掩盖效应？

同一批 227 个已观察事件，在同一份日期制执行契约下各排一次日程——一次从申报接受时间起，
一次从最早 SEC 披露起——用同一批已保存的价格与公司行动重算发行人和 SPY 的现金留存收益，
唯一的差别是时钟；再在同一完整样本上运行冻结的发行人聚类推断（10,000 次 bootstrap）。

忠实性检查：182 个窗口未变的事件，重算收益与冻结账本完全一致（最大差 0.0）。11 个事件因
披露时钟入场早于已保存的价格窗口而排除，同一样本为 **216 个事件 / 48 家**，其中 90 个入场
提前（1–24 个交易日，中位数 3）。

### 结果

四种表示与 20 日市场超额收益的 Spearman IC（见上表）在两种时钟下区间全部跨零；最大变化是
Innovation 的 +0.03，只有区间宽度的四分之一。**时钟修正没有改变结论。**

两个附带子集（因为算了所以报告，不是预注册的）：Innovation 在披露时真正可知的 97 个事件上，
IC 在两种时钟下都约 +0.04（区间 [−0.15, +0.25]）；入场提前的 90 个事件上，Level IC 从 −0.037
变为 −0.165（区间 [−0.346, +0.024]），这是已观察结果上的子集观察，不是证据。

### 怎么读

- 发布时点修正对**有效性**是必要的，但在这批样本上不改变答案：两种时钟下毛利率的任何表示都
  不能预测 20 日超额收益。
- 227 个事件里有 130 个的 Innovation 用到了发布之后、定期报告之前才到达的信息——这是关于
  原研究有效性的发现，与收益无关。按披露时钟整体重放预测是下一步。
- 这些结果早已观察过，本诊断只能诊断时钟，不能选模型、不算验证。

### 边界

假设毛利率在发布时已公开（在 27 个抽查数值中 26 个一致，未逐事件认证）；两种时钟都是 SEC
代理；未计算成本、行业中性化、策略或盈亏。
