# Prospective Consensus Membership Review v1

## English

### Question and boundary

Does an aggregate consensus change require retained analysts to revise their
estimates, or could net analyst entry/exit alone explain it? This is an
outcome-free diagnostic, not an estimate of actual analyst behavior. It reads
only the frozen quality ledger and preserved consensus snapshots. No actual
revenue, Innovation, prices, returns or model fitting is used.

The review has no admission authority. Original `quality_status` and
`downstream_admission` are preserved; historical warnings do not disappear when
a later snapshot has an unchanged count. Review completion is not issue
resolution. No quarantined or review-required event is released.

### Conditional arithmetic

Write the old/new count and mean as n0, n1, m0, m1. Under an unweighted
arithmetic-mean assumption, the change in the sum of forecasts is:

```text
D = n1*m1 - n0*m0
implied entering/leaving mean = D / (n1 - n0)
```

For net additions, test that implied mean against the current high/low range.
For removals, test it against the previous range. Conditional bounds on the
average retained-analyst revision are:

```text
addition of k: [(D - k*current_high)/n0, (D - k*current_low)/n0]
removal of k:  [(D + k*previous_low)/n1, (D + k*previous_high)/n1]
```

The exported percentage bounds divide these dollar changes by m0. They are
counterfactual identification bounds, not empirically estimated revisions,
analyst surprise or forecast-error sigma. Relative tolerance 1e-10 is numerical,
not an economic materiality threshold.

The smaller panel is assumed to be a subset of the larger panel, with no
simultaneous additions and removals. Retained estimates are assumed unchanged
for the composition-only test. Under that assumption an enlarged panel cannot
have a higher minimum or lower maximum; a reduced panel cannot have a lower
minimum or higher maximum. Range monotonicity is tested separately.

All of these are necessary conditions, not sufficient conditions. Aggregate
data do not verify unweighted averaging, panel nesting, identical membership,
or whether the high/low and count refer to exactly the same contributors.
Unchanged counts also do not rule out churn. Rejection can indicate genuine
revisions, churn, different aggregation or provider changes; it does not prove
which explanation is correct. Analyst-level IDs, forecasts and timestamps, plus
provider mapping history, would be needed for actual attribution.

### 2026-09-14 result

The immutable review covers all 37 registered events and 148 consecutive
snapshot pairs, including nine review-required and three quarantined events.
For the latest comparisons of the nine review-required events:

- eight do not rule out composition-only changes under the necessary bounds;
- AAPL 2026-09-30 rejects the simple nested composition-only explanation;
- no event's actual attribution is identified and no event is promoted.

Ford 2026-09-30 illustrates the issue: its mean rose by 1.4169% while coverage
rose from seven to eight analysts. Under the conditional model, adding an
estimate of USD 51.287 billion, equal to the current high estimate, can account
for the entire mean increase without any retained analyst changing a forecast.
This is a possible counterfactual, not a verified account of what happened.

The nine review-required and three quarantined events remain blocked. The
current admitted capacity is 25 events from 14 issuers, below the preregistered
30-event / 15-issuer minimum. Capacity is not return readiness: even an adequate
cohort still needs original actuals, complete price/action evidence, and both
Innovation tails. With unresolved quality issues, the original cohort cannot
meet the current formal minimum. This is a research-feasibility warning, not a
failed signal-effectiveness verdict. No threshold is reduced and no frozen
event is replaced or appended.

The September 1–14 pairs contain 12 uncaptured intervening calendar dates.
Changes are interval-censored between the two real observations. Today's data
cannot recover the historical path or be backdated into those dates.

### Run and inspect

The daily runner automatically appends this diagnostic after the quality audit;
decisions still consume the original quality ledger, never the review ledger.

```powershell
uv run python -m scripts.build_prospective_consensus_review `
  --quality-root <quality-root> `
  --readiness-root <matching-readiness-root>
```

Parent report/table hashes are verified before use. Outputs include a paired
Parquet counterfactual table with both source hashes and clocks, a 37-event
review table, a JSON lineage report, and `consensus_membership_review.md`.
Running the same inputs reuses the immutable output. Unit tests use synthetic
snapshots and need no API key or downloaded data.

## 中文

### 做了什么

用每次真实快照的平均收入预期、人数和高低范围，检验平均值变化是否可能仅由分析师加入或退出
造成。复核覆盖全部37个冻结事件、148个配对观察，不读取实际收入、收益或模型结果。

9个待审事件中，8个不能排除“仅成员变化”这个条件性解释；AAPL九月财季不符合这个简单解释。
例如Ford上调约1.42%、人数7→8，仅加入一位预测51.287十亿美元的分析师，在算术上就可能产生
全部变化。我们没有逐分析师身份和预测，所以不能认定这就是事实，也不能据此解除待审。

平均值必须是人数对应的简单算术平均，高低值必须属于同一个集合；还假设只发生净加入或净退出、
没有同时换人。这些假设未被聚合数据验证。人数不变也可能换人；简单解释被拒绝也不等于证实了
真实上修、供应商错误或其他某个具体原因。报告清楚区分“均值层面未排除”和范围单调性矛盾。

### 结果及约束

全部原始质量规则和准入状态保持不变：25个干净、9个待审、3个隔离，解除数为0。当前最多可准入
25个事件、14家发行人，低于冻结正式检验门槛30个事件、15家发行人。若问题无法凭证据解决，原队列
无法达到门槛，但这不代表信号已经被证明无效。也不能为了得到结果而降低门槛或替换失败事件。

9月1日到14日之间有12个未采的日历日期。我们只知道变化发生在两个真实快照之间，无法还原某天
具体发生了什么，绝不将今天的数据倒填。下一步需要讨论逐分析师证据或独立、事前注册的新队列；
本轮未购买数据、未扩样、未解封，也未生成交易订单。
