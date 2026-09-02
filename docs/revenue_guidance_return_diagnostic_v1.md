# Revenue-Guidance Innovation: First Return Diagnostic

## English

### Executive conclusion

The project has established a useful expectation model, but it has not established a tradable
return signal. On the frozen 35-event forecast sample, point-in-time management guidance predicts
reported revenue much more accurately than the prior-year same-quarter baseline. After an
outcome-blind market-data expansion, 23 events from eight issuers have authorized 20-session
market-excess returns. In those 23 observations, the rank correlation between the raw
revenue-guidance Innovation and subsequent excess return is approximately zero.

This is an informative diagnostic, not a failed project. It separates two questions that are often
conflated:

1. Is management guidance a good forecast of the later accounting result? **Yes, in the selected
   exact-quarter-guidance sample.**
2. Is actual revenue minus a months-old management forecast a good representation of the market's
   surprise immediately before the filing? **Not established.**

The second result suggests that a good accounting expectation is not automatically the right market
expectation. Prices and analysts can update between the guidance date and the target filing.

### Research design

The atomic observation is Company × Filing Event × Information Set. The frozen expectation sample
contains 35 events from 13 issuers. Every management-guidance observation comes from a prior SEC
filing accepted before the target actual became available. The target-quarter revenue, guidance
horizon, metric, unit, accession, acceptance timestamp, and raw-response hash are retained.

The raw Innovation is:

> target-quarter reported revenue − midpoint of the previously disclosed exact-quarter revenue
> guidance range

It is also expressed as a percentage of the absolute guidance midpoint. This is a
management-guidance forecast error. It is not labelled analyst surprise and is not yet standardized.
A standardized value remains forbidden until strictly earlier out-of-sample guidance errors provide
a defensible forecast-error sigma.

Market evidence was added only after the 35-event forecast sample and its gates were frozen. The
return contract is:

- entry: second trading-day unadjusted open after the SEC acceptance timestamp;
- exit: unadjusted close on the twentieth holding session;
- outcome: issuer total return minus SPY total return, arithmetically;
- controls: official exchange sessions, security-identity continuity, corporate-action evidence,
  delisting outcomes, and a pre-event median dollar-volume threshold of USD 1 million; and
- selection: no return value is used to authorize, exclude, or replace an event.

Exact vendor payloads and hashes are preserved locally. Licensed event-level prices are not intended
for public redistribution.

### Coverage audit

| Stage | Events | Issuers | Status |
|---|---:|---:|---|
| Frozen guidance forecasts | 35 | 13 | Forecast sample fixed before returns |
| Authorized return complete cases | 23 | 8 | 65.71% event coverage |
| Positive raw Innovations | 20 | — | Tail gate passes |
| Negative raw Innovations | 3 | — | Tail gate fails |

Twelve forecast events remain outside the formal return ledger. Nine lack sufficient historical
pre-event liquidity evidence, primarily because free symbol-based history is unavailable after
acquisition, delisting, or data-vendor symbol purging. Three have reconstructed histories but fall
below the frozen USD 1 million liquidity threshold. These events were not replaced after observing
returns. Consequently, the 23-event diagnostic is conditional on the evidence-complete, liquid
subset and must not be described as survivorship-bias-free full-market evidence.

### Results

The expectation-model result remains strong:

| Forecast diagnostic | Result |
|---|---:|
| Complete forecast sample | 35 events / 13 issuers |
| Management-guidance MAPE | 2.81% |
| Prior-year same-quarter MAPE | 13.73% |
| Relative MAPE improvement | 79.53% |
| Guidance event win rate | 77.14% |
| Issuer-cluster bootstrap 95% interval for APE difference | [-15.29, -6.78] pp |

The new return observations are descriptive only:

| Return diagnostic | Observed value |
|---|---:|
| Complete return cases | 23 events / 8 issuers |
| Spearman rank correlation | -0.0069 |
| Pearson correlation | +0.0025 |
| Mean 20-session market-excess return | +1.14% |
| Fraction with positive excess return | 52.17% |
| Positive-Innovation group | 20 events; mean -0.04%; median -5.12% |
| Negative-Innovation group | 3 events; mean +9.03%; median +11.75% |

The apparent reversal between the positive and negative groups must not be interpreted as a short
signal. The negative group contains only three events and the full sample only eight issuers.

### Statistical gate and honest interpretation

The preregistered return gate requires at least 20 events, 10 issuers, five positive Innovations,
and five negative Innovations. The event-count and positive-tail requirements pass; issuer breadth
and the negative tail fail. Therefore:

- the observed correlations and group summaries are persisted as **non-inferential descriptives**;
- the formal return-statistics artifact remains empty;
- no issuer-cluster confidence interval or p-value is produced; and
- no alpha, profitability, or signal-validity claim is permitted.

The point estimate nevertheless changes the research direction. Management guidance is a strong
baseline for forecasting the accounting number, but its forecast error is not necessarily the
market innovation at filing time. The economically correct state should be the expectation
immediately before the event, not merely the most recent management number disclosed months earlier.

### Next experiment

The next frozen comparison should be:

1. **Seasonal expectation:** prior-year same-quarter revenue.
2. **Management expectation:** latest eligible management guidance.
3. **Contemporaneous market expectation:** analyst consensus available immediately before the
   filing, with timestamp, analyst count, dispersion, staleness, and revision history.

The primary new Innovation should be actual revenue minus the as-of analyst consensus. Management
guidance remains a state feature and forecast baseline. Recent consensus revisions, pre-event price
momentum, volume, volatility, and—where economically justified—options-implied information can be
added as state features, but none should be selected by inspecting the 23 observed returns.

Before formal inference, the project should also recover point-in-time histories for acquired and
delisted securities from a source that retains dead symbols, expand issuer breadth and the negative
surprise tail under the same frozen rules, and preregister transaction-cost and subgroup diagnostics.
Deep learning is not warranted at the current sample size; transparent regularized or tree-based
baselines should follow only after the information set and labels are defensible.

---

## 中文

### 核心结论

项目已经建立了一个有价值的 expectation 模型，但还没有建立一个可交易的收益信号。在冻结的
35 个事件上，严格点时的管理层季度收入指引，比去年同季度收入更准确地预测了之后公布的收入。
在不查看收益的前提下补齐市场证据后，目前 8 家发行人的 23 个事件具有合规的 20 个交易日市场
超额收益。在这 23 个观察中，raw revenue-guidance Innovation 与后续超额收益的秩相关约为零。

这不是项目失败，而是把两个经常被混在一起的问题分开了：

1. 管理层指引是否能较好预测后来的会计实际值？**在有精确同季度指引的样本中，可以。**
2. “实际收入减去几个月前的管理层预测”是否代表财报公布前一刻的市场 surprise？**尚未成立。**

第二个结果说明：好的会计预测基准，不一定就是正确的市场预期基准。从管理层发布指引到目标财报
公布之间，价格和分析师预期还会持续更新。

### 研究设计

研究原子单位是 Company × Filing Event × Information Set。冻结的 expectation 样本包含
13 家发行人的 35 个事件。每条管理层指引都来自目标实际值可得之前已被 SEC 接受的 filing；
系统保留目标财季、收入指标、单位、accession、接受时间和原始响应哈希。

Raw Innovation 定义为：

> 目标季度实际收入 − 此前公开的同季度收入指引区间中点

同时用指引中点的绝对值进行比例化。它是“管理层指引预测误差”，不是分析师 surprise，目前也
不是 Standardized Innovation。只有积累足够严格更早的样本外指引误差后，才允许用历史预测误差
形成 sigma。

只有在 35 个预测事件和预测 Gate 完全冻结后，系统才补充市场结果。收益契约为：

- SEC 接受时间后的第二个交易日，以未调整开盘价进入；
- 第 20 个持有交易日，以未调整收盘价退出；
- 结果为发行人总收益减 SPY 总收益；
- 使用正式交易所日历、证券身份连续性、公司行动、退市结果，并要求事件前中位日成交额至少
  100 万美元；
- 任何事件的授权、排除或替换都不读取收益数值。

原始供应商响应与哈希保存在本地；受许可约束的事件级行情不计划在公开仓库重新分发。

### 覆盖审计

| 阶段 | 事件 | 发行人 | 状态 |
|---|---:|---:|---|
| 冻结的管理层指引预测 | 35 | 13 | 查看收益前已经固定 |
| 授权收益完整样本 | 23 | 8 | 事件覆盖率 65.71% |
| 正向 raw Innovation | 20 | — | 尾部 Gate 通过 |
| 负向 raw Innovation | 3 | — | 尾部 Gate 未通过 |

仍有 12 个预测事件未进入正式收益账本：9 个缺少充分的事件前流动性历史，主要因为免费按代码查询
的历史数据会在并购、退市或供应商清除旧代码后消失；另外 3 个虽然重建了价格历史，但低于冻结的
100 万美元流动性门槛。我们没有在看到收益后替换这些公司。因此，这 23 个事件只代表“证据完整且
具备流动性”的条件子样本，不能宣称为已经完全消除幸存者偏差的全市场证据。

### 结果

Expectation 模型的正面结果保持不变：

| 预测诊断 | 结果 |
|---|---:|
| 完整预测样本 | 35 个事件 / 13 家发行人 |
| 管理层指引 MAPE | 2.81% |
| 去年同季 MAPE | 13.73% |
| MAPE 相对改善 | 79.53% |
| 管理层指引逐事件胜率 | 77.14% |
| APE 差的发行人聚类 bootstrap 95% 区间 | [-15.29, -6.78] pp |

新加入的收益结果目前只能作描述：

| 收益诊断 | 观察值 |
|---|---:|
| 完整收益样本 | 23 个事件 / 8 家发行人 |
| Spearman 秩相关 | -0.0069 |
| Pearson 相关 | +0.0025 |
| 平均 20 日市场超额收益 | +1.14% |
| 超额收益为正的比例 | 52.17% |
| 正向 Innovation 组 | 20 个；均值 -0.04%；中位数 -5.12% |
| 负向 Innovation 组 | 3 个；均值 +9.03%；中位数 +11.75% |

正负两组表面上的反转不能被解释成做空信号，因为负向组只有 3 个事件，全部样本也只有 8 家发行人。

### 统计 Gate 与诚实解释

预注册收益 Gate 要求至少 20 个事件、10 家发行人、5 个正向 Innovation 和 5 个负向 Innovation。
当前事件数和正向尾部通过，但发行人广度与负向尾部未通过。因此：

- 相关系数和分组结果只保存为**非推断性描述统计**；
- 正式收益统计表仍为空；
- 不计算发行人聚类置信区间和 p 值；
- 不允许宣称 alpha、盈利能力或信号已经有效。

尽管如此，点估计已经足以调整研究路线：管理层指引是预测会计实际值的强基准，但其预测误差不一定
是财报时点的市场 Innovation。经济上更合理的 State，应该是事件发生前一刻的预期，而不是几个月前
最后一次公开的管理层数字。

### 下一轮实验

下一轮应冻结比较三种 expectation：

1. **季节性预期：** 去年同季度收入；
2. **管理层预期：** 最近一次符合条件的管理层指引；
3. **当期市场预期：** 财报公布前一刻可得的分析师一致预期，并保留时间戳、分析师数量、分歧度、
   新鲜度和历史修订路径。

新的主要 Innovation 应定义为“实际收入减去 as-of 分析师一致预期”。管理层指引保留为 State Feature
和预测基准。近期一致预期修订、事件前价格动量、成交量、波动率，以及在经济逻辑充分时的期权隐含
信息，可以加入 State Features，但不能根据当前 23 个收益结果筛选变量。

正式推断之前，还需要从能保留死亡代码的数据源恢复被并购和退市证券的点时历史，在相同冻结规则下
扩大公司广度和负向 surprise 尾部，并预注册交易成本与子样本诊断。当前样本量不适合深度学习；只有
当信息集和标签可靠后，才应进入透明的正则化模型或树模型。
