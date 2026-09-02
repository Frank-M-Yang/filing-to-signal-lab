# Prospective Market-Innovation v3 Preregistration

## English

### Why v3 is a new hypothesis

The external temporal holdout rejected both the historical Conditional Drift baseline and its
Innovation–Reaction Alignment extension. Those 16 outcomes are now permanently observed and may
never be used to select another candidate.

V3 does not tune the failed model. It changes the expectation being measured:

> **Raw Market Innovation = reported revenue / last analyst revenue consensus observable before
> SEC acceptance − 1.**

The hypothesis is that post-filing drift is related to information that was unexpected relative to
the market's immediately pre-filing expectation. The primary score has no return-fitted parameter.

### Frozen prospective cohort

On 2026-08-29, before any registered target actual or return existed, the system froze 37 future
company-quarter events from 20 issuers. The periods run from 2026-08-31 through 2027-01-31. All seven
registration gates passed. The registry admits no replacement after a missing filing, provider
failure, delisting, or adverse outcome.

The previous 16-event holdout is stored in a separate firewall. O1 and O3 are explicitly retired;
neither their features nor their scores may enter v3.

### Level / Delta / Innovation experiment

All representations will be compared on one complete-case Company × Filing Event × Information Set
sample:

| Representation | Frozen definition | Role |
|---|---|---|
| State | Analyst-implied revenue growth versus the strictly prior-year same-quarter actual | Same-sample benchmark |
| Delta | Last eligible consensus versus the earliest eligible consensus snapshot | Same-sample benchmark |
| Raw Innovation | Reported revenue versus the last eligible pre-acceptance consensus | Primary signal |
| Standardized Innovation | Raw Innovation divided by only strictly earlier out-of-sample consensus errors | Secondary signal |
| Reaction | Market-excess reaction observable before the second-session-open entry | Diagnostic only |

The analyst high–low range is a dispersion State feature, not a forecast-error sigma. Reaction is
not part of the primary v3 score.

### Frozen evaluation rules

The consensus forecast must first improve on the prior-year same-quarter baseline. Return evidence
then requires at least 30 complete events, 15 issuers, eight positive Innovations, and eight
negative Innovations. The primary estimand is Spearman rank correlation between Raw Innovation and
the net 20-session market-excess return. Entry is the second trading-day open after SEC acceptance;
exit is the twentieth-session close. The cost proxy is computed from pre-acceptance data.

Issuer-cluster bootstrap inference, leave-one-issuer-out stability, and top-minus-bottom tertile
returns are frozen before outcomes. No annualized return, portfolio backtest, investment advice, or
live-trading authorization is currently in scope.

### Current status

- Registered: 37 events / 20 issuers / six target period-ends.
- Initial capture depth: one date.
- Registered actual values loaded: zero.
- Registered returns loaded: zero.
- External validation metrics: not computed.

The registry-aware readiness ledger covers 37/37 registered events and reuses one audited SEC
submissions request per issuer to monitor all six period-ends. Its collection stop is the event's
future SEC acceptance—not the fiscal period end. This matters because the most informative analyst
revision may occur after the quarter closes but before the company reports. The current foundation
gate passes with 20 SEC issuers covered; revision depth remains zero because only one independent
capture date exists.

The next evidence-producing action is repeated daily consensus capture, followed by future SEC
filing detection. Empty Innovation and return fields are the correct current result.

### Reproduction

```powershell
uv run python -m scripts.freeze_prospective_market_innovation_v3
uv run python -m scripts.build_prospective_market_innovation_readiness
uv run python -m scripts.run_forward_research_daily
```

The first command is idempotent for the frozen input manifests. The daily command must not be used
to alter the registered cohort or protocol.

---

## 中文

### 为什么 v3 是新的经济假设

时间外留出检验已经否定 Conditional Drift 基线及其 Innovation–Reaction Alignment 扩展。
那 16 个结果永久属于已观察样本，今后不能再拿来挑选候选模型。

v3 不继续调失败模型，而是改变 expectation 的定义：

> **Raw Market Innovation = 实际披露收入 ÷ SEC 接受前最后一次可验证的分析师收入一致预期 − 1。**

研究假设是：财报后的漂移与“相对于财报前一刻市场预期的新信息”有关。主信号不含任何根据
收益拟合的参数。

### 已冻结的前瞻样本

2026-08-29，在所有目标实际值和收益出现之前，系统冻结了 20 家发行人的 37 个未来公司—财季
事件，目标期从 2026-08-31 延伸到 2027-01-31，七项注册 Gate 全部通过。未来即使发生数据源
失败、退市或不利结果，也不得替换公司或事件。

之前 16 个留出事件被写入独立防火墙；O1 和 O3 被明确退休，其特征与分数不得进入 v3。

### Level / Delta / Innovation 实验

所有表示必须在同一套完整 Company × Filing Event × Information Set 样本上比较：

| 表示 | 冻结定义 | 角色 |
|---|---|---|
| State | 分析师隐含收入增长，相对于严格点时的去年同季度实际值 | 同样本基准 |
| Delta | 最后一次合格一致预期相对于最早合格快照的变化 | 同样本基准 |
| Raw Innovation | 实际收入相对于公告前最后一次合格一致预期的偏离 | 主信号 |
| Standardized Innovation | Raw Innovation 除以严格更早的样本外共识误差波动 | 次要信号 |
| Reaction | 第二交易日开盘前已经可观察的市场超额反应 | 仅诊断 |

分析师预期高低区间只是分歧 State，不能冒充预测误差 sigma；Reaction 不进入 v3 主信号。

### 冻结的验证规则

首先，一致预期必须在预测误差上优于去年同季度基准。随后收益检验至少需要 30 个完整事件、
15 家发行人、8 个正向 Innovation 和 8 个负向 Innovation。主统计量是 Raw Innovation 与成本后
20 日市场超额收益的 Spearman 秩相关。建仓时点固定为 SEC acceptance 后第二个交易日开盘，
退出为第 20 个交易日收盘；成本代理只使用公告前数据。

发行人聚类 bootstrap、逐一剔除发行人稳定性和 Top–Bottom 三分位收益差均已在结果出现前冻结。
当前不计算年化收益、不做组合回测、不形成投资建议，也不授权实盘。

### 当前状态

- 已注册：37 个事件、20 家发行人、6 个目标期末。
- 初始快照深度：1 个采集日。
- 已加载目标实际值：0。
- 已加载目标收益：0。
- 外部有效性指标：尚未计算。

注册表驱动的就绪账本现已覆盖 37/37 个事件，并复用每家发行人的一次 SEC submissions 请求来
监控全部 6 个目标期末。采集截止点是未来对应财报的 SEC acceptance，而不是财季结束日；因为
最有信息量的一致预期修订很可能发生在财季结束后、正式财报发布前。当前 20 家 SEC 公司均已
覆盖，基础采集 Gate 通过；但只有 1 个独立采集日，所以 Revision 深度仍为零。

下一项真正增加证据的工作是持续保存每日一致预期，并等待未来 SEC 财报。现在 Innovation 和收益
保持为空，正是严格时间顺序下应有的结果。

### 复现

```powershell
uv run python -m scripts.freeze_prospective_market_innovation_v3
uv run python -m scripts.build_prospective_market_innovation_readiness
uv run python -m scripts.run_forward_research_daily
```

第一个命令对相同输入 manifest 是幂等的；日常流水线不得修改已经冻结的样本或协议。
