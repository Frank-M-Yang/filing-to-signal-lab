# Tradeable Dataset and Baseline Alpha Tournament v1

## English

### Status

The implementation is operational, but the current verdict is
**`insufficient_unobserved_validation_evidence`**. This is an intentional research gate, not a
software failure and not evidence for or against tradable alpha.

The 58 event outcomes in the existing longitudinal study had already been inspected before this
model round began. They are therefore frozen as `legacy_observed_training_only`: they may help
fit a future walk-forward model once enough new events arrive, but they can never be counted as
validation evidence. The sealed holdout begins at `2026-08-28T00:00:00Z` and is not evaluated by
the v1 tournament.

### Frozen question

On the same complete-case filing events, does adding point-in-time filing information improve
post-filing return forecasts beyond information already present in recent market prices?

The comparison is deliberately split into:

- **Filing-only:** Level, Delta, Raw Innovation, Standardized Innovation, and form type;
- **Market-only:** pre-filing momentum, volatility, volume, market trend, relative momentum, and
  beta;
- **Combined:** the exact union of both feature sets on the same event sample.

This design prevents a larger or easier filing sample from being mistaken for incremental
financial-statement value.

### Point-in-time data contract

The atomic row is Company × Filing Event × Information Set. The primary label is the arithmetic
market-excess return from the second trading-day open after SEC acceptance through the twentieth
holding-session close. The five-session return is retained only as a secondary label.

Each row records three separate clocks:

1. `feature_cutoff`: SEC filing acceptance time;
2. `execution_entry_at`: the formal tradable entry;
3. `target_available_at_20d`: the time at which the primary outcome becomes observable.

They must satisfy `feature_cutoff < execution_entry_at < target_available_at_20d`. Market
features use only sessions strictly before the filing calendar date. This is conservative for
after-hours filings, but it prevents ambiguous same-day closes from leaking into the feature
vector. Pre-event closes are adjusted for historical splits; future bars and future corporate
actions cannot enter the calculation.

### Current local build

| Stage | Events | Issuers |
|---|---:|---:|
| Frozen primary event ledger | 58 | 15 |
| Filing features complete | 49 | 15 |
| Market features complete | 42 | 15 |
| Combined features and primary label complete | 35 | 15 |
| Unobserved validation candidates | 0 | 0 |
| Sealed holdout candidates | 0 | 0 |

Sixteen events lack the required 21 strictly prior market sessions. The other sample losses come
from the previously frozen filing-analysis eligibility and the same-sample intersection. No
missing market feature is imputed in v1.

### Baseline tournament

Seven deterministic candidates are frozen before new validation outcomes arrive:

| ID | Algorithm | Feature set |
|---|---|---|
| M0 | Zero-return benchmark | None |
| M1 | Ridge | Filing-only |
| M2 | Ridge | Market-only |
| M3 | Ridge | Combined |
| M4 | Histogram gradient boosting | Filing-only |
| M5 | Histogram gradient boosting | Market-only |
| M6 | Histogram gradient boosting | Combined |

Validation is quarterly walk-forward. For every validation quarter, a training row is eligible
only when its 20-session outcome was already observable before the quarter began. Random splits,
same-time labels, and sealed-holdout rows are forbidden. All seven models score the same events.

The default research gate requires at least 2,000 training events from 100 issuers and 500
unobserved validation events across eight eligible quarters. A selected combined model must have
a mean quarterly Spearman IC of at least 0.02 and exceed the best market-only model by at least
0.01. Passing these gates would authorize a separate paper-trading design review, not live
trading. The code always reports `paper_trading_authorized=false` and
`live_trading_authorized=false` in v1.

### Reproduction

These commands require the local licensed market-data and processed SEC layers; those artifacts
are intentionally excluded from Git:

```powershell
uv run python -m scripts.build_tradeable_dataset
uv run python -m scripts.run_baseline_alpha_tournament
```

The first command writes an immutable preregistration registry, feature store, feature-clock
audit, and sample funnel. The second writes fold audits, walk-forward predictions, model
diagnostics, and selection checks. Every output manifest contains SHA-256 links to its input
manifests.

### Interpretation and next gate

The positive result of this round is that the repository now has a model-ready contract that
refuses to recycle already-seen outcomes as proof. The current zero-prediction tournament result
means that genuine model evaluation has not started yet.

The next data round should add new, previously uninspected events across more issuers, industries,
size buckets, and failed or delisted securities. Filing text, analyst expectations, guidance,
options, short interest, and macro-regime features should be added only as independently timed
feature families. The Filing-only, Market-only, and Combined comparison must remain unchanged
until the first unobserved validation gate is reached.

---

## 中文

### 当前状态

系统已经可以运行，但当前判定是 **`insufficient_unobserved_validation_evidence`**，即
“尚无足够的未观察验证证据”。这是研究协议主动拦截，不是程序失败，也不是证明信号有效
或无效。

上一轮纵向研究的 58 个事件收益已经被我们看过，因此在这一轮永久标记为
`legacy_observed_training_only`。未来它们可以在严格走步训练中作为历史训练材料，但永远
不能冒充新验证样本。封存测试集从 `2026-08-28T00:00:00Z` 开始，v1 不读取其评价指标。

### 冻结的研究问题

在完全相同的财报事件样本上，加入点时财报信息，能否比近期市场价格中已经包含的信息更好
地预测财报后收益？

系统把输入明确拆成三组：

- **仅财报：** Level、Delta、Raw Innovation、Standardized Innovation 和报表类型；
- **仅市场：** 公告前动量、波动率、成交量、市场趋势、相对动量和 beta；
- **组合：** 在同一完整样本上合并上述两组特征。

这样可以避免把“财报样本更多或更容易”误认为财报信息真的带来了增量预测价值。

### 点时数据契约

每一行的原子单位是 Company × Filing Event × Information Set。主要标签是从 SEC 接受
公告后的第二个交易日开盘，到第 20 个持有交易日收盘的算术市场超额收益；5 日收益只保留
为次要标签。

每个事件记录三只独立的时钟：

1. `feature_cutoff`：SEC 接受 filing 的时间；
2. `execution_entry_at`：正式可交易的建仓时间；
3. `target_available_at_20d`：主要结果真正可观察的时间。

三者必须严格满足“特征截止 < 建仓 < 结果可得”。市场特征只使用公告日期之前的交易日，
即便公司盘后公告也不使用当天收盘，以保守地消除时点歧义。公告前价格会按历史拆股调整；
未来价格和未来公司行动不能进入特征。

### 当前本地构建结果

| 阶段 | 事件数 | 发行人数 |
|---|---:|---:|
| 冻结的主要事件账本 | 58 | 15 |
| 财报特征完整 | 49 | 15 |
| 市场特征完整 | 42 | 15 |
| 组合特征和主要标签完整 | 35 | 15 |
| 从未看过结果的验证候选 | 0 | 0 |
| 封存测试候选 | 0 | 0 |

其中 16 个事件没有满足 21 个严格早于公告的市场交易日历史。其他损耗来自上一轮冻结的
财报分析资格和共同样本交集；v1 不对缺失市场特征进行填补。

### 基线模型竞赛

在任何新验证结果到来之前，预先冻结七个候选：零收益基准，以及 Ridge 和直方图梯度提升
分别使用“仅财报”“仅市场”“组合”三套特征。

验证采用按季度向前走步。每个验证季度开始时，训练事件的 20 日结果必须已经真实可得；
禁止随机切分、同期标签和封存测试集进入训练。七个模型必须评价完全相同的事件。

默认研究门槛要求至少 2,000 个训练事件、100 家发行人，以及分布在至少 8 个季度的 500
个全新验证事件。入选组合模型的季度平均 Spearman IC 至少达到 0.02，而且必须比最好的
仅市场模型高至少 0.01。即使通过，也只允许进入独立的模拟盘设计评审，不等于批准实盘。
v1 始终明确写出 `paper_trading_authorized=false` 和 `live_trading_authorized=false`。

### 如何运行

以下命令需要本地持牌行情和处理后的 SEC 数据；这些数据按许可证和隐私边界不会上传 Git：

```powershell
uv run python -m scripts.build_tradeable_dataset
uv run python -m scripts.run_baseline_alpha_tournament
```

第一条命令生成不可变的旧结果登记表、特征库、特征时钟审计和样本漏斗；第二条生成走步
预测、折叠审计、模型诊断和选择门槛。每个输出 manifest 都通过 SHA-256 绑定全部上游版本。

### 结果含义与下一道门

这一轮真正的正面成果，是项目现在拥有一套会主动拒绝“拿看过的答案再次证明自己”的
训练数据契约。当前模型产生 0 条验证预测，代表真正的模型有效性评价尚未开始。

下一轮应该扩展从未观察过结果的新事件，覆盖更多公司、行业、市值层级，以及失败和退市
证券。公告文本、分析师预期、公司指引、期权、卖空和宏观状态可以逐组加入，但每组都必须
有独立的点时时钟。在达到首个未观察验证门槛之前，“仅财报 / 仅市场 / 组合”的比较规则
不再修改。
