# Point-in-Time Management Guidance as an Expectation Model

> **Update (2026-08-28):** The outcome-blind market-data expansion increased authorized return
> coverage from 12 to 23 events. The frozen inferential gate still fails because the sample contains
> only eight issuers and three negative Innovations. See
> `docs/revenue_guidance_return_diagnostic_v1.md` for the current return diagnostic. The forecast
> findings below are unchanged.

## English

### Research question

The original Level/Delta/Innovation study showed that a mechanically fitted Innovation can remain
mostly a re-expression of Delta. This stage asks a narrower, falsifiable question:

> Does management's exact-quarter revenue guidance, disclosed before the target filing, forecast
> target-quarter revenue more accurately than the strictly point-in-time prior-year same-quarter
> value?

This is a forecast experiment, not a return backtest. Return outcomes were not loaded when the
expectation model was selected.

### Point-in-time design

The atomic observation remains Company × Filing Event × Information Set. The target registry
contains 2,924 2022–2025 filing events. Of these, 1,445 events from 110 issuers have a comparable
USD revenue actual. One event per issuer-calendar-year was selected by a deterministic SHA-256
rank that does not use revenue magnitude, guidance, or returns, producing 385 target events.

For each target, the system searched the preceding 30–240 days for an accepted 8-K, preferring
Item 2.02 and falling back to Item 7.01. It found 372 prior 8-K candidates. Exact response bytes,
URLs, acceptance timestamps, and SHA-256 hashes were preserved before parsing.

Guidance was admitted only when all of the following were explicit:

- the statement was forward-looking rather than a current-result or preliminary-result sentence;
- the horizon was the same fiscal quarter as the target;
- the metric was GAAP-compatible revenue;
- the unit belonged to that revenue statement rather than another number nearby;
- a point or bounded range could be converted to USD without inference; and
- the guidance acceptance timestamp preceded the target actual's availability timestamp.

Thirty-five events from 13 issuers passed this complete-case contract. That is 9.41% of the 372
candidate events. The low coverage is an important property of the result, not a parser failure to
be hidden: many firms do not provide exact-quarter revenue guidance, and many disclosures provide
annual, non-GAAP, qualitative, or ambiguous guidance.

### Forecast tournament

Two expectations were compared on exactly the same 35 events:

1. **Seasonal baseline:** the prior fiscal year's same-quarter revenue, available before the
   guidance filing.
2. **Management guidance:** the midpoint of the prior exact-quarter revenue guidance range.

No model was fitted. The preregistered gate required at least 30 events, 10 issuers, 10% relative
MAPE improvement, a 60% event win rate, and an issuer-cluster bootstrap 95% upper bound below zero
for the event-level APE difference (guidance minus seasonal). The bootstrap used 5,000 draws with
seed 41.

### Results

| Diagnostic | Result |
|---|---:|
| Complete cases | 35 events / 13 issuers |
| Management-guidance MAPE | 2.81% |
| Seasonal-baseline MAPE | 13.73% |
| Relative MAPE improvement | 79.53% |
| Guidance event win rate | 77.14% |
| Mean APE difference, guidance minus seasonal | -10.92 percentage points |
| Issuer-cluster bootstrap 95% interval | [-15.29, -6.78] percentage points |
| One-sided cluster-bootstrap p-value | 0.00020 |
| Worst leave-one-issuer-out improvement | 76.11% |

All forecast gates passed. The result does not appear to be driven by one issuer: after excluding
each issuer in turn, the relative MAPE improvement remains between approximately 76.1% and 81.1%.

### Interpretation

This is the first strong positive empirical result in the expectation-model branch of the
project. It supports the project's core intuition: the relevant state immediately before an
event is not always well represented by a historical level. A contemporaneous, previously
disclosed expectation can provide a much cleaner reference point for the event innovation.

The result is conditional. It applies to the selective subset of firms and quarters for which an
exact, comparable revenue guidance range exists. It does not establish that guidance is available
for the full market, that management guidance is unbiased in all regimes, or that the resulting
revenue surprise predicts a tradable return.

### Return evidence boundary

Only after the forecast gate passed did a separate return-readiness audit join the 35 forecasts to
the outcome-blind authorized return cohort. Twenty-three events from eight issuers now have a
complete 20-session market-excess return using the second trading-day unadjusted open and the
twentieth-session unadjusted close. Twenty Innovations are positive and three are negative.

The frozen return gate requires at least 20 events, 10 issuers, and at least five events on each
side of zero. It therefore fails on issuer breadth and the negative tail. The observed Spearman
correlation of -0.0069 is persisted only as a non-inferential descriptive; the formal statistics
table remains empty and no confidence interval, p-value, or alpha claim is reported. Raw and
standardized innovations are also kept separate; standardized guidance Innovation is withheld
until strictly earlier out-of-sample guidance errors provide a defensible sigma history.

### Next evidence-producing step

Add a contemporaneous, as-of market-expectation layer—preferably timestamped analyst consensus and
its revision path—and recover dead-symbol histories from a source that retains acquired and
delisted securities. Expand issuer breadth and the negative surprise tail under the same frozen
rules. The next test should compare seasonal, management-guidance, and immediately pre-event market
expectations on the same events rather than tune the signal on the observed returns.

### Reproduction commands

```powershell
$env:SEC_USER_AGENT="Researcher Name contact@example.com"
uv run python -m scripts.build_revenue_guidance_expansion --collect --limit 500
uv run python -m scripts.run_revenue_guidance_tournament
uv run python -m scripts.build_revenue_guidance_return_audit
uv run streamlit run app.py
```

The SEC collection command is credential-free but requires an identified User-Agent. The final
return audit also requires the locally reconstructed, authorized market-return ledger; licensed
market payloads are not distributed in Git.

---

## 中文

> **更新（2026-08-28）：** 无结果选样的市场数据扩展，已把授权收益覆盖从 12 个事件提高到
> 23 个事件。由于目前只有 8 家发行人和 3 个负向 Innovation，冻结的正式推断 Gate 仍未通过。
> 当前收益诊断见 `docs/revenue_guidance_return_diagnostic_v1.md`；下述预测结论不变。

### 研究问题

原来的 Level / Delta / Innovation 研究发现，机械拟合出来的 Innovation 仍可能主要是
Delta 的重新表达。这一阶段把问题收窄成一个可以被证伪的实验：

> 在目标财报公告之前，管理层已经公开的同季度收入指引，能否比严格点时的去年同季度
> 收入更准确地预测目标季度收入？

这是预测实验，不是收益回测。选择 expectation 模型时没有加载股票收益。

### 点时研究设计

原子样本仍是 Company × Filing Event × Information Set。目标登记表包含 2022–2025 年
2,924 个 filing 事件，其中 110 家发行人的 1,445 个事件拥有可比的美元收入实际值。系统
不看收入大小、指引或收益，只按固定 SHA-256 排名，在每个发行人—日历年中选择一个事件，
得到 385 个目标事件。

对每个目标，系统在之前 30–240 天内寻找已经被 SEC 接受的 8-K，优先 Item 2.02，缺失时
才使用 Item 7.01，最终得到 372 个既往 8-K 候选。解析前保存原始响应字节、URL、接受时间和
SHA-256 哈希。

只有同时满足下列条件的收入指引才可进入研究：

- 语句明确面向未来，而不是本期结果或 preliminary result；
- 指引期限明确对应目标财季；
- 指标是可与 GAAP 收入比较的 revenue；
- 单位属于该收入语句，不能借用附近另一项指标的 million/billion；
- 点预测或区间可以不靠主观推断地换算成美元；
- 指引接受时间严格早于目标实际值的可得时间。

最终有 13 家发行人的 35 个事件通过完整样本契约，占 372 个候选事件的 9.41%。这个低覆盖
本身就是结论的一部分：很多公司不提供精确的同季度收入指引，另一些只提供全年、非 GAAP、
定性或期限含混的指引。

### 预测竞赛

在完全相同的 35 个事件上比较两种 expectation：

1. **季节性基准：** 指引公告前已经可得的去年同财季收入。
2. **管理层指引：** 既往同季度收入指引区间的中点。

这一轮没有拟合模型。预注册 Gate 要求：至少 30 个事件、10 家发行人、MAPE 相对改善至少
10%、逐事件胜率至少 60%，并且“管理层指引 APE − 季节性 APE”的发行人聚类 bootstrap
95% 区间上界小于零。Bootstrap 固定为 5,000 次、seed 41。

### 结果

| 诊断指标 | 结果 |
|---|---:|
| 完整样本 | 35 个事件 / 13 家发行人 |
| 管理层指引 MAPE | 2.81% |
| 去年同季基准 MAPE | 13.73% |
| MAPE 相对改善 | 79.53% |
| 管理层指引逐事件胜率 | 77.14% |
| 平均 APE 差，指引减季节性 | -10.92 个百分点 |
| 发行人聚类 bootstrap 95% 区间 | [-15.29, -6.78] 个百分点 |
| 单侧聚类 bootstrap p 值 | 0.00020 |
| 留一发行人检验中的最差改善 | 76.11% |

全部预测 Gate 均通过。结果也不是由某一家发行人单独造成：依次剔除每一家发行人后，MAPE
相对改善仍约在 76.1%–81.1% 之间。

### 结果代表什么

这是项目 expectation-model 分支目前第一个较强的正面实证结果。它支持项目最核心的直觉：
事件发生前的“状态”不一定能由某个历史 level 代表；此前已经公开、且更贴近当前状态的预期，
可以成为计算 Innovation 的更干净基准。

但这个结论是有条件的。它只适用于主动提供精确、同季度、可比收入指引的公司与季度；不能
外推为全市场都能获得管理层指引，也不能证明管理层在所有时期都无偏，更不能直接证明由此
得到的收入 surprise 可以赚钱。

### 收益证据边界

只有预测 Gate 通过之后，独立的收益就绪审计才把 35 个预测事件与无收益选样的授权收益样本
连接。目前 8 家发行人的 23 个事件拥有完整的 20 日市场超额收益：公告后第二个交易日以未调整
开盘价进入，第 20 个交易日以未调整收盘价退出。其中正向 Innovation 20 个，负向只有 3 个。

冻结的收益 Gate 要求至少 20 个事件、10 家发行人，而且零点两侧各至少 5 个事件，因此当前在
发行人广度和负向尾部上没有通过。观察到的 Spearman 相关为 -0.0069，但只保存为非推断性描述；
正式统计表仍为空，不计算置信区间、p 值或宣称 alpha。Raw Innovation 与 Standardized
Innovation 也保持分离；只有积累足够严格更早的样本外指引误差后，才允许形成用于 sigma 的历史。

### 下一项真正增加证据的工作

增加一个严格 as-of 的当期市场预期层，优先使用带时间戳的分析师一致预期和修订路径；同时从能
保留死亡代码的数据源恢复被并购和退市证券的历史。在相同冻结规则下扩大公司广度和负向 surprise
尾部。下一轮应在同一批事件上比较季节性预期、管理层指引和公告前一刻的市场预期，而不是根据
已经看到的收益调参。

### 复现命令

```powershell
$env:SEC_USER_AGENT="Researcher Name contact@example.com"
uv run python -m scripts.build_revenue_guidance_expansion --collect --limit 500
uv run python -m scripts.run_revenue_guidance_tournament
uv run python -m scripts.build_revenue_guidance_return_audit
uv run streamlit run app.py
```

SEC 抓取不需要付费 Key，但必须使用可识别的 User-Agent。收益审计还需要在本地重建并授权的
市场收益账本；有许可约束的事件级行情不会随 Git 分发。
