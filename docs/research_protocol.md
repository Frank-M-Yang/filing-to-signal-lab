# Research Protocol: State, Change, and Innovation

## Primary question

> **Do markets price financial states, changes in states, or unexpected innovations
> in states? And how much do ex-post revisions distort that conclusion?**

中文：

> 市场究竟在定价企业所处的状态、状态的变化，还是相对于事前预期的新信息？使用事后
> 修订数据又会怎样扭曲这个结论？

项目不以发现更多财务因子为目标。第一项正式研究只使用毛利率，把
`Level → Delta → Forecast → Innovation → Reaction → Drift` 一条链做成严格的
point-in-time 实验。

## Three falsifiable questions

### Q1 — Representation

Level、Delta、Innovation 哪种表示更能解释公告后的异常收益？

### Q2 — Efficiency

信息主要在首次可执行价格发现窗口中被吸收，还是在后续 20 个交易日继续漂移？

### Q3 — Data integrity

使用 latest/restated 数据替代当时可得的原始数据，会把上述结论改变多少？

这三个问题都允许空结果。若三种表示在严格点时数据上都没有稳定关系，应报告没有证据，
而不是调整定义直至获得正向收益。

## Atomic observation

研究的原子单位是：

```text
Company × Filing Event × Information Set
```

一行 `filing_event_panel` 表示某家公司在某次财报事件发生时，市场已知什么、模型在事件前
预测了什么、披露带来什么创新，以及市场随后如何反应。

代码契约位于 `event_panel.py`。核心字段分为：

- 身份：`event_id`, `company_id`, `ticker`, `sector`；
- 会计期间：`fiscal_year`, `fiscal_quarter`, `period_end`；
- 信息时钟：`filing_accepted_at`, `actual_available_at`, `executable_at`；
- PIT facts：`revenue`, `gross_profit`, `gross_margin`, `gm_yoy_lag`；
- State：`gm_level`, `gm_delta_yoy`；
- Expectation：`gm_forecast`, `forecast_feature_cutoff`,
  `forecast_training_end`, `forecast_generated_at`；
- Innovation：`gm_innovation_raw`, `gm_innovation_std`；
- 标准化血缘：`forecast_sigma`, `forecast_error_history_end`,
  `forecast_sigma_observations`；
- 原始血缘：`revenue_fact_id`, `gross_profit_fact_id`, `filing_accession`；
- 事后比较：`gross_margin_latest`, `latest_available_at`, `was_revised`,
  `revision_size`；
- 明确的结果窗口及异常收益。

`latest_*` 字段只允许用于 Q3，不允许成为 PIT 预测或标准化的输入。

## Representations

对同一个原始毛利率事实定义：

```text
Level(i,t) = GM(i,t)
Delta(i,t) = GM(i,t) - GM(i,t-4)
RawInnovation(i,t) = GM(i,t) - Forecast(i,t-)
StdInnovation(i,t) = RawInnovation(i,t) / Sigma(i,t-)
```

Delta 属于状态层，描述企业状态的历史变化；Innovation 属于事件层，描述新观察相对于
事件前条件预期的差异。第一轮实验同时报告 raw 和 standardized innovation。若只有
standardized innovation 有效，需要把预测误差本身与不确定性估计的贡献分开解释。

## Forecast specification

第一版固定使用 expanding-window pooled OLS，不允许自动选择变量：

```text
GM(i,t) = beta0
        + beta1 * GM(i,t-4)
        + beta2 * DeltaGM(i,t-4)
        + beta3 * RevenueGrowth(i,t-1)
        + beta4 * IndustryGM(i,t-1)
        + error(i,t)
```

`IndustryGM(i,t-1)` 是明确滞后的行业信息。第一版禁止直接计算当前季度完整横截面的行业
中位数，因为尚未披露的同行会造成泄漏。未来可以单独增加严格 as-of 的
`PeerInformation(i,t-)` 实验。

每个事件必须满足：

```text
forecast_feature_cutoff < actual_available_at
forecast_training_end < actual_available_at
forecast_generated_at < actual_available_at
forecast_training_end <= forecast_generated_at
forecast_feature_cutoff <= forecast_generated_at
```

同一时间戳到达的事件作为一个批次预测，不能使用彼此的实际值。模型预测完成后，当前
实际值才能进入后续事件的训练集。

默认需要至少 200 个历史训练观察。具体阈值可以在预注册或数据审计后调整，但不能根据
收益结果选择。

## Standardization without leakage

标准化分母只能来自当前事件之前已经产生的历史 **out-of-sample** 预测误差：

```text
forecast_error_history_end < actual_available_at
```

当前事件的误差、同一时间戳的同行误差、in-sample residual 和未来误差都不能进入分母。
默认至少需要 100 个历史 OOS errors。项目测试包括“改变未来实际值不能改变过去预测或
sigma”以及“同一时间戳同行不能互相泄漏”。

## Outcome windows

不使用含义模糊的 `[0,+1]` 或 `[+2,+20]` 作为唯一字段。每行保存精确 UTC 时间和价格
类型：

### Immediate reaction

```text
reaction_start = filing event 之前最后一个市场收盘
reaction_end   = filing event 之后第一个可执行交易日收盘
```

用于研究价格发现。这个窗口通常包含不可在公告前建仓捕获的反应，不应描述为可交易收益。

### Tradable drift

```text
tradable_start = 第二个可执行交易日开盘
tradable_end   = 第 20 个交易日收盘
```

用于研究公告后是否存在潜在可实施的延迟反应。实际适配器必须通过交易所日历、公告时间和
盘前/盘后状态生成这些时间戳，不能只加普通工作日。

## Common sample and normalization

每次 Level、Delta、Innovation 比较必须限制在共同样本：

```text
S = events where Level, Delta, Innovation, and the selected outcome all exist
```

代码提供 `select_common_research_sample`，防止三种表示因数据覆盖不同而不可比。

第一版在每个研究横截面使用完全相同的标准化方法。若样本量允许，主要规格使用行业内
median/MAD，普通 z-score 作为稳健性检查。标准化方法和最小横截面数量必须在查看收益
结果前固定。

## Experiments

### Experiment 1 — Representation

在相同 PIT 共同样本上比较：

| Test | Level | Delta | Raw innovation | Standardized innovation |
|---|---:|---:|---:|---:|
| Rank IC — reaction | ? | ? | ? | ? |
| Rank IC — drift | ? | ? | ? | ? |
| Q5–Q1 — reaction | ? | ? | ? | ? |
| Q5–Q1 — drift | ? | ? | ? | ? |
| Quantile monotonicity | ? | ? | ? | ? |
| Positive-IC year share | ? | ? | ? | ? |
| Industry stability | ? | ? | ? | ? |

第一项正式研究不把 Sharpe 或年化收益作为主要结果。

### Experiment 2 — PIT versus latest

在同一事件和同一结果窗口上形成 3×2 矩阵：

| Representation | PIT data | Latest/restated data |
|---|---:|---:|
| Level | L(PIT) | L(Latest) |
| Delta | Delta(PIT) | Delta(Latest) |
| Innovation | I(PIT) | I(Latest) |

定义：

```text
RevisionInflation = Performance(latest) - Performance(PIT)
```

这里的 Performance 必须明确指某个已注册指标，例如 mean Rank IC 或 Q5–Q1，而不是混合
多个表现统计。

## Interpretation rules

- Innovation 只解释 reaction：市场快速吸收信息，没有明显可交易 drift；
- Innovation 同时解释 drift：存在与创新相关的公告后延迟反应，仍需交易成本和稳健性检验；
- latest 有效但 PIT 无效：历史数据版本泄漏可能制造了结论；
- standardized 有效但 raw 无效：结果可能依赖不确定性估计，应单独审计 sigma；
- 三者均无效：没有证据支持毛利率表示具有稳定预测力。

## Current status and evidence boundary

已实现：

- 规范化事件面板列和 point-in-time 验证器；
- 冻结的 82 家核心资产，以及 2,298 个带 accession、接受时间与 fact 血缘的真实季度事件；
- 1,954 个严格 Level 事件和 1,723 个严格 Delta 事件；
- 813 个 expanding-window OOS Raw Innovation 和 713 个带严格先验 sigma 的共同表示事件；
- 毛利率 Level/Delta 派生；
- expanding-window pooled OLS OOS 预测；
- strictly-prior OOS error sigma；
- raw/standardized innovation；
- 共同样本选择；
- Apple 2009 修订案例，用于说明 Q3 的机制。
- v1 表示诊断：预测 RMSE 仅较去年同季改善约 3.5%，Raw Innovation 与 Delta 的
  Spearman 相关约 0.88；Innovation 已可审计构造，但尚未形成足够独立的表示。
- `ExpectationAdjustment = GM(t-4) - Forecast(t-)` 诊断：其波动约为 Delta 的 30%，
  42.9% 的事件跨五分位边界，但只有 6.2% 移动两个或更多分位；当前模型主要进行相邻排序
  调整。
- v1 预测稳定性审计：OOS R² 为 6.9%，公司×年份聚类的 MAE、RMSE 与 OOS R² 诊断区间
  均跨过零；六个至少有25个预测的年份里，MAE 改善年份为0。
- v2 选模顺序和主指标已预注册：MAE 为主，RMSE/OOS R² 为辅；distinctness 只作为结果
  报告，不作为优化目标。
- v2 已在 812 个 M0–M4 共同事件上执行。M2 company-aware OLS 的 MAE skill 为 35.4%，
  公司×年份聚类区间为 21.0% 至 46.3%，并在六个合格年份全部为正；M2 因此通过财务预测
  gate。M3 也通过但未超过 M2，M4 仅作为非线性稳健性模型。

尚未实现：

- 历史行业分类和 historical security master；
- 对 12 家会计口径或经济适用性异常公司的人工复核；
- 正式交易所日历、点时价格、退市收益和异常收益；
- 82 家核心资产的真实收益实验。

因此当前项目已有真实 PIT 披露数据和防泄漏计算基础设施，但尚无价格结果证据回答 Q1 或 Q2。
模拟回测仍只是偏差机制演示，不得作为研究结论。

V2 的固定候选、指标与资格 gate 见
[Forecast tournament v2 preregistration](forecast_tournament_v2_preregistration.md)，执行结果见
[forecast tournament v2 results](forecast_tournament_v2_results.md)。M2 通过的是财务预测 gate；
由于历史股票池、历史 SIC 和价格数据 gate 仍未满足，仍不能进入正式收益测试。
