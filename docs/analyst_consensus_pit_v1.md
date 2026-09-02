# Point-in-Time Analyst Consensus Layer v1

## English

### Why this layer exists

The first revenue-guidance return diagnostic produced a useful separation: management guidance is
a strong forecast of the later accounting result, but actual revenue minus months-old guidance has
approximately zero rank correlation with the currently available 20-session excess returns. A
plausible reason is that the market updates its expectation between the guidance date and the target
filing.

The next primary representation is therefore:

> **Market Innovation = reported revenue − analyst revenue consensus available immediately before
> the filing**

Management guidance remains a State feature and an accounting-forecast baseline. It is not silently
relabelled as analyst surprise.

### Frozen sample and clock

The consensus collection plan was created from the already frozen 35 revenue-guidance events from
13 issuers. It uses the historical security identity for every event and does not load a return
value. The plan contains:

- SEC target accession and acceptance-time-derived cutoff;
- historical ticker and exchange identity;
- target fiscal year and exact quarter;
- deterministic query and target identifiers;
- secret-free provider URLs; and
- explicit flags prohibiting provider-coverage selection and post-failure replacement.

For every event, `consensus_cutoff_at` equals the target actual's SEC availability timestamp. A
provider record is admitted to the formal model only if its own version timestamp is strictly
earlier than this cutoff. A record obtained today that merely contains a historical consensus value
does not satisfy the contract by itself.

### Provider assessment

| Source | Useful fields | Point-in-time conclusion | Role |
|---|---|---|---|
| Massive / Benzinga Earnings | historical earnings event, estimated revenue, currency, GAAP/adjusted method, `last_updated` | Testable per record; reject if `last_updated >= cutoff` | First structured pilot; separate paid entitlement |
| Alpha Vantage Earnings Estimates | quarterly revenue consensus, high/low, analyst count; EPS revision fields | Historical rows do not expose a record-version/as-of timestamp in the observed schema | Forward-only daily snapshots for future events; no historical backfill |
| Business Quant Estimates | historical consensus and actuals | Documentation does not establish a versioned as-of snapshot for each historical estimate | Candidate for coverage audit, not yet formal |
| Eulerpool PIT Estimates | explicitly described as point-in-time estimate snapshots | Promising, but schema, history, licensing, and event coverage must be tested | Candidate second provider |
| I/B/E/S PIT / Bloomberg / Capital IQ | institutional historical estimate versions and broad dead-security coverage | Appropriate formal benchmark if licensed | Long-run research source |

The observed provider probes closed two tempting but invalid shortcuts. The Massive/Benzinga
request authenticated, but the account lacked the separately licensed Earnings entitlement.
Eulerpool returned 200 estimate rows for two tickers, but every observed row was EPS-only and the
available `as_of` dates were current snapshots rather than the historical pre-filing revenue
versions required by this experiment. The evidence-backed provider gate therefore remains closed.

### Forward-only ledger

Rather than backfill history with values whose historical availability cannot be proven, the
repository now freezes a deterministic 20-company cohort from the existing 82-company core-asset
universe. A daily Alpha Vantage capture stores the exact response bytes, SHA-256, fetch timestamp,
provider URL with the secret removed, and quarterly revenue average/high/low/analyst count. The
collector timestamp is the availability clock. These observations are eligible only for filings
that occur after capture; every row carries `historical_backtest_admission = false`.

Twenty calls per capture fit inside the standard 25-request daily allowance and leave five calls of
operational headroom. The cohort is selected before observing future returns and is not replaced
after provider failures.

The first capture on 2026-08-29 completed all 20 targets without a provider failure. It preserved
752 quarterly observations, including 37 strict forward-period candidates across all 20 issuers.
The strict gate requires at least one target period on or after the capture date for every issuer;
historical rows returned in the same response remain snapshot context and are not admitted as
historical backtest evidence.

Massive documents the Benzinga Earnings endpoint as a separately licensed USD 99/month individual
expansion with history from April 2010. The endpoint provides a consensus revenue value but not the
analyst count or consensus dispersion. Those missing fields remain unavailable; they are not
imputed.

### Admission statuses

Every provider result must end in one of the following evidence states:

- `point_in_time_consensus_ready`: exact ticker, fiscal year, quarter, USD GAAP revenue, event-date
  alignment, and record version strictly before the SEC cutoff;
- `retrospective_only_post_event_or_unverifiable_version`: a plausible historical consensus exists,
  but its point-in-time availability is not proven;
- a specific metric, date, identity, or fiscal-period rejection; or
- `provider_query_pending` / entitlement failure.

Only the first status can enter the formal Level/Delta/Innovation experiment. Retrospective records
may be used to diagnose potential coverage but never to claim a tradable historical signal.

### Frozen next experiment

After the provider gate reaches at least 20 events and 10 issuers, compare three expectations on the
same complete-case events:

1. prior-year same-quarter revenue;
2. latest eligible management guidance; and
3. immediately pre-filing point-in-time analyst consensus.

The primary return test is prespecified for analyst-consensus Innovation. Forecast accuracy is
reported for all three expectations, but no representation is selected by inspecting returns. The
existing 23 return outcomes remain closed until the consensus sample and transformation rules are
frozen.

### Commands

Build the target plan and an audit from already preserved bytes:

```powershell
uv run python -m scripts.build_revenue_consensus
```

After loading an authorized Massive key into the current process, run a one-event entitlement and
schema probe:

```powershell
uv run python -m scripts.build_revenue_consensus --collect --limit 1
```

The key is read only from `MASSIVE_API_KEY`; it is never written to a query URL, manifest, or report.

Build the forward-only cohort and audit saved snapshots:

```powershell
uv run python -m scripts.build_alpha_vantage_consensus
```

After loading an Alpha Vantage key into the current process, capture today's 20 snapshots:

```powershell
uv run python -m scripts.build_alpha_vantage_consensus --collect --limit 20
```

The key is read only from `ALPHA_VANTAGE_API_KEY`. It is stripped from response URLs before any
manifest is written. Current snapshots are never relabelled as historical point-in-time evidence.

---

## 中文

### 为什么要增加这一层

第一轮收入指引收益诊断把两个问题分开了：管理层指引可以很好地预测之后的会计实际值，但
“实际收入减去几个月前的指引”与当前 20 日超额收益的秩相关约为零。一个合理解释是，在指引日
与目标财报之间，市场已经持续更新了预期。

所以下一项主要表示定义为：

> **Market Innovation = 公布收入 − 财报前一刻可得的分析师收入一致预期**

管理层指引继续作为 State Feature 和会计预测基准，不能被偷偷改称为 analyst surprise。

### 冻结样本与信息时钟

一致预期抓取计划来自已经冻结的 13 家发行人、35 个收入指引事件。它使用每个事件的历史证券
身份，不加载任何收益数值。计划保存：

- 由 SEC 目标 accession 和接受时间确定的 cutoff；
- 历史 ticker 与交易所身份；
- 目标财年和精确季度；
- 确定性的 query ID 与 target ID；
- 不包含密钥的供应商 URL；
- 禁止按供应商覆盖率选样、禁止抓取失败后替换公司的标记。

每个事件的 `consensus_cutoff_at` 都等于目标实际值在 SEC 的可得时间。只有供应商记录自身的
版本时间严格早于 cutoff，才能进入正式模型。今天查询到一条“历史 consensus 数字”，本身不能
证明该数字当时已经可得。

### 数据源判断

| 数据源 | 有用字段 | 点时结论 | 用途 |
|---|---|---|---|
| Massive / Benzinga Earnings | 历史财报事件、收入预期、币种、GAAP/调整口径、`last_updated` | 可以逐条检验；`last_updated >= cutoff` 时拒绝 | 第一版结构化试验；需要独立付费权限 |
| Alpha Vantage Earnings Estimates | 季度收入共识、高低值、分析师数量；EPS 修订字段 | 已观察的历史行没有记录版本/as-of 时间 | 对未来事件做每日前向快照；禁止历史回填 |
| Business Quant Estimates | 历史一致预期和实际值 | 文档尚不能证明每条历史估计都有版本化 as-of 快照 | 可做覆盖审计，暂不正式使用 |
| Eulerpool PIT Estimates | 明确称为点时 estimates snapshots | 有希望，但仍需测试字段、历史、许可和覆盖 | 第二候选供应商 |
| I/B/E/S PIT / Bloomberg / Capital IQ | 机构级历史估计版本和更完整的死亡证券覆盖 | 获得许可后适合作为正式基准 | 长期研究数据源 |

供应商实测排除了两个看似省事、实际不满足研究时钟的捷径。Massive/Benzinga 请求通过了身份
验证，但账户没有单独付费的 Earnings 权限。Eulerpool 对两个 ticker 返回了 200 行 estimates，
但实测全部只有 EPS 字段；可见的 `as_of` 也是当前快照日期，而不是目标财报之前的历史收入
版本。因此，证据化的正式供应商 Gate 仍然关闭。

### 前向数据账本

与其用无法证明历史可得性的数字回填，项目现在从既有 82 家核心资产中确定性冻结 20 家公司，
每天保存一次 Alpha Vantage 快照。每次采集保留原始响应字节、SHA-256、抓取时间、去除密钥的
供应商 URL，以及季度收入平均值、高低值和分析师数量。抓取时间就是可得性时钟。这些记录只
允许匹配抓取之后发生的财报；每行都明确标记 `historical_backtest_admission = false`。

每次 20 个请求低于标准免费额度的每日 25 次，并留出 5 次运维余量。公司名单在未来收益出现
之前选定，供应商抓取失败后也不替换样本。

2026-08-29 的首次采集完成了 20/20 个目标，没有供应商失败，共保存 752 条季度记录。其中 37 条
属于明确的前向财季候选，覆盖全部 20 家公司。严格 Gate 要求每家公司至少有一个目标财季不早于
抓取日；同一响应中的历史行只保留为快照上下文，不能进入历史回测。

Massive 官方把 Benzinga Earnings 列为单独的每月 99 美元个人扩展，历史从 2010 年 4 月开始。
该端点提供一致收入预期，但不提供分析师数量和一致预期分歧度；这些缺失字段会保持缺失，不做填补。

### 准入状态

每条供应商结果必须落入以下证据状态之一：

- `point_in_time_consensus_ready`：ticker、财年、季度、美元 GAAP 收入、事件日期均匹配，且记录
  版本严格早于 SEC cutoff；
- `retrospective_only_post_event_or_unverifiable_version`：存在看似合理的历史一致预期，但无法
  证明当时可得；
- 明确的指标、日期、身份或财季不匹配；
- `provider_query_pending` 或权限失败。

只有第一种状态可以进入正式 Level/Delta/Innovation 实验。Retrospective 记录只能用于判断潜在
覆盖率，不能用来宣称可交易的历史信号。

### 已冻结的下一项实验

当供应商 Gate 达到至少 20 个事件、10 家发行人后，在完全相同的完整样本上比较：

1. 去年同季度收入；
2. 最近一次符合条件的管理层指引；
3. 财报前一刻严格点时的分析师一致预期。

分析师一致预期 Innovation 是预先指定的主要收益检验。三种 expectation 都报告预测准确度，
但不能查看收益后再选择表示。现有 23 个收益结果必须继续封存，直到一致预期样本和变换规则冻结。

### 运行命令

只生成目标计划并审计已经保存的原始字节：

```powershell
uv run python -m scripts.build_revenue_consensus
```

把有权限的 Massive Key 加载到当前进程后，先做一次单事件权限与字段试验：

```powershell
uv run python -m scripts.build_revenue_consensus --collect --limit 1
```

密钥只从 `MASSIVE_API_KEY` 读取，不会写入查询 URL、manifest 或报告。

生成前向公司清单并审计已经保存的快照：

```powershell
uv run python -m scripts.build_alpha_vantage_consensus
```

把 Alpha Vantage Key 加载到当前进程后，抓取今天的 20 家快照：

```powershell
uv run python -m scripts.build_alpha_vantage_consensus --collect --limit 20
```

密钥只从 `ALPHA_VANTAGE_API_KEY` 读取，所有 manifest 落盘前都会从响应 URL 中移除密钥。
今天的快照绝不会被重新标成历史点时证据。
