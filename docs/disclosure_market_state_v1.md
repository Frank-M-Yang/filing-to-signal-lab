# Disclosure-Cut Market State v1

## English

### Question

The v2 market-state view (`validation_market_state`) computes 13 "pre-filing"
features from sessions strictly before the New York date of the 10-Q/10-K
acceptance. The [first-public clock](first_public_clock_v1.md) shows that an
Item 2.02 earnings release often came earlier. Did those pre-filing features
already contain the price reaction to the release?

### Method

- Same inputs, same code: the frozen v2 module is recomputed, not modified,
  and the disclosure view calls the identical feature function. Only the cutoff
  clock differs: the earliest SEC disclosure instead of the filing.
- Two behaviour-preserving extractions in `validation_market_state.py` share
  input loading and market-input preparation between the views; all 19
  existing v2 tests pass unchanged.
- Events join the clock table by `(cik, target accession)`. The clock table must
  reproduce the cohort's filing clock exactly, and an earliest disclosure may
  never follow its filing; either failure stops the build.
- The earliest disclosure is an **event clock only**. Financial values are not
  made available earlier (`financial_value_clock_changed = false`).

### Results

Validation cohort: 227 events, 48 issuers. All 227 reports are in the clock
table: 218 have an earlier Item 2.02 release, 9 do not. The v2 view reproduces
its known 175 market-state-ready events.

| Quantity | Events |
|---|---:|
| Cutoff date moved earlier | 101 |
| v2 ready | 175 |
| Disclosure view ready | 166 |
| **v2 used at least one close after the earliest disclosure** | **58 (25 issuers)** |
| — still ready, features changed | 49 |
| — no longer ready (fewer than 61 strictly prior sessions) | 9 |
| Changed only by the conservative date cut (no post-release close) | 20 |
| Ready in both, unchanged | 97 |

For the 49 affected events that stay ready, a median of 6 sessions is removed
(maximum 24). The absolute change in 5-day momentum has median 3.9 percentage
points and 90th percentile 14.2; the 5/20-day volume ratio changes by a median
0.20. The 9 events that lose readiness were ready in v2 only because of
post-release sessions.

### Interpretation

- About a third of the v2-ready events (58/175) carried part of the earnings
  reaction inside features labelled pre-filing state.
- Relative to a 10-Q-based entry these prices were observable, so this is not
  look-ahead for that trading rule; it is a mislabelled state variable. Any
  model that treated these features as pre-event state conflated state with
  reaction.
- Once the event is defined at the earliest disclosure, the v2 features become
  genuine look-ahead for those events and must not be used.
- Existing consumers (tradeable dataset, conditional-drift features, admission
  ledgers) still read v2. Switching them is a separate, versioned decision.

### Boundaries

- Clocks are SEC proxies, not certified first-public times; an 8-K may be
  furnished up to four business days after the announcement.
- The New York date cut is conservative: it also drops a pre-release session on
  the release date.
- No outcome is loaded and no return is computed; admission gates stay closed.

### Reproduce

```powershell
uv run python -m scripts.build_disclosure_market_state --clock-root data/processed/first_public_clock/<bundle>
uv run pytest tests/test_disclosure_market_state.py tests/test_validation_market_state.py
```

The synthetic tests check that the disclosure view excludes post-release prices,
is invariant to any price after the release, reproduces v2 exactly when there is
no earlier release, and keeps events without a clock under an explicit status.

## 中文

### 问题与方法

v2 市场状态视图用 10-Q/10-K 接受时间所在纽约日期之前的交易日计算 13 项"事件前"特征。
[首次公开时钟](first_public_clock_v1.md)表明财报 8-K 往往更早。本视图使用**同一批输入和同一个
特征函数**，只把截止时钟换成最早的 SEC 披露时间；冻结的 v2 模块只做了两处不改变行为的函数提取，
19 个原有测试全部通过。首次披露只作为**事件时钟**，不会让财务数值提前可用。

### 结果

验证队列 227 个事件、48 家发行人，全部找到时钟（218 个有更早的财报 8-K）。v2 视图复现了已知的
175 个就绪事件。其中 **58 个事件（25 家发行人）在 v2 中用到了财报发布之后的收盘价**：49 个在新视图中
仍然就绪但特征全部改变，5 日动量变化的中位数为 3.9 个百分点、90 分位为 14.2 个百分点；另外 9 个
失去就绪状态，因为它们在 v2 中之所以够 61 个交易日，本来就依赖发布后的交易日。另有 20 个事件仅因
保守的日期截止而改变，97 个事件不变。

### 含义

- 约三分之一的 v2 就绪事件，其"事件前状态"特征里混入了财报反应。
- 相对按 10-Q 入场的规则，这些价格在入场前可见，所以不是该规则下的前视泄漏，而是**状态变量被错标**；
  但一旦事件按首次披露定义，继续使用 v2 特征就是真正的前视泄漏。
- 现有下游（可交易数据集、条件漂移特征、准入台账）仍读取 v2，是否切换需要单独的版本化决定。
- 未加载结果、未计算收益，准入门保持关闭。
