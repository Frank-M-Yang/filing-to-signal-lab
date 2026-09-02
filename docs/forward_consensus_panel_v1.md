# Forward-Only Analyst Consensus Panel v1

## English

### Research purpose

Historical analyst-consensus values are unsafe unless their historical availability can be
proved. The project therefore starts a prospective ledger: preserve what the provider returned
today, attach the collector timestamp, and use the observation only for events that become
available later.

The panel separates three objects:

- **Level:** the current average quarterly revenue estimate;
- **Revision Delta:** the change in that estimate between two distinct capture dates for the same
  ticker and fiscal-period end; and
- **Innovation:** future reported revenue minus the latest eligible consensus snapshot captured
  strictly before the SEC filing acceptance timestamp.

Consensus dispersion is descriptive state information, calculated as `(high − low) / average`.
It is not an expected return.

### First capture

The frozen cohort contains 20 active issuers selected deterministically from the existing
82-company core-asset universe without loading returns. The first Alpha Vantage capture on
2026-08-29 completed 20/20 targets with no provider failure.

- 752 total quarterly observations were preserved with exact response bytes and SHA-256 hashes;
- 37 target periods were on or after their capture date and therefore qualified as strict
  forward-period candidates;
- all 20 issuers have at least one strict forward candidate;
- forward revenue averages and analyst counts have no missing or non-positive values; and
- analyst counts range from 4 to 52 in the preserved snapshot.

The remaining historical rows are retained as snapshot context only. Every row is marked
`historical_backtest_admission = false`.

### Current gates

| Gate | Result | Meaning |
|---|---:|---|
| Future-event watchlist | Pass | All 20 issuers have a strict forward revenue expectation |
| Distinct capture dates | 1 / 2 | A single date cannot define a revision |
| Issuers with revisions | 0 / 10 | No cross-date Delta exists yet |
| Innovation | Blocked | The future SEC actual is not available |
| Return research | Not opened | No return value is loaded or used for selection |

This is a positive data-integrity milestone, not evidence of alpha. The system now has one genuine
pre-event market-expectation state. It does not yet have a time series, an accounting surprise, or
a trade outcome.

### Reproduction

Capture one daily snapshot per issuer with an Alpha Vantage key loaded only in the current process:

```powershell
uv run python -m scripts.build_alpha_vantage_consensus --collect --limit 20
uv run python -m scripts.build_forward_consensus_panel
uv run streamlit run app.py
```

The collector is resumable within a UTC date. Re-running it on the same date reuses completed
targets rather than creating false intraday revisions. The Streamlit view is **Result 07 ·
Forward Consensus Ledger**.

### Next evidence

The next distinct-date capture can create the first Revision Delta. When a watched company later
files, the actual-value join must match the frozen CIK and fiscal period and select only the latest
consensus snapshot with `captured_at < SEC accepted_at`. Price and return windows remain closed
until that join is complete.

The companion SEC monitor baseline now covers all 20 issuers. It preserves 616 eligible historical
10-Q/10-K observations, all accepted before the consensus snapshot, and correctly reports zero
future filing candidates. See [Forward SEC Filing Monitor v1](forward_filing_monitor_v1.md).

---

## 中文

### 研究目的

如果无法证明一条历史分析师预期在当时已经可得，就不能安全地拿它做历史回测。因此项目改为
建立前向账本：保存供应商今天实际返回的内容，记录抓取时间，只允许它服务于之后才发生的事件。

面板严格区分三个对象：

- **Level：** 当前季度收入一致预期的平均值；
- **Revision Delta：** 同一 ticker、同一财季在两个不同抓取日期之间的预期变化；
- **Innovation：** 未来实际披露收入，减去 SEC 接受时间之前最后一条合格一致预期。

一致预期分歧度定义为 `(最高预期 − 最低预期) / 平均预期`，它只是市场预期状态，不是预期收益。

### 首次采集

观察名单从既有 82 家核心资产中确定性选出 20 家活跃发行人，选样过程不读取收益。2026-08-29
首次 Alpha Vantage 采集完成 20/20 家，没有供应商失败。

- 保存 752 条季度记录，并保留原始响应字节和 SHA-256；
- 37 个目标财季不早于其抓取日期，属于严格前向候选；
- 20 家公司全部至少拥有一个严格前向候选；
- 前向收入平均预期和分析师数量没有缺失或非正值；
- 已保存快照中的分析师数量范围为 4 至 52。

供应商同时返回的历史行只保留为快照上下文，每行都标记
`historical_backtest_admission = false`。

### 当前 Gate

| Gate | 结果 | 含义 |
|---|---:|---|
| 未来事件观察名单 | 通过 | 20 家公司都有严格前向收入预期 |
| 不同抓取日期 | 1 / 2 | 单个日期无法定义预期修订 |
| 具有 Revision 的公司 | 0 / 10 | 尚不存在跨日 Delta |
| Innovation | 阻止 | 未来 SEC 实际值尚未公布 |
| 收益研究 | 尚未开启 | 没有加载收益，也没有用收益选样 |

这是一个正面的数据完整性里程碑，但不是 alpha 证据。我们现在拥有一个真实的事件前市场预期
状态，还没有预期时间序列、会计 surprise 或交易结果。

### 运行方式

把 Alpha Vantage Key 只加载到当前进程后，每家公司每天保存一次快照：

```powershell
uv run python -m scripts.build_alpha_vantage_consensus --collect --limit 20
uv run python -m scripts.build_forward_consensus_panel
uv run streamlit run app.py
```

采集器在同一个 UTC 日期内可恢复执行；当天重复运行会复用已完成目标，不会制造假的日内修订。
Streamlit 中选择 **成果 07 · 前向一致预期账本** 即可查看。

### 下一项证据

下一个不同日期的快照可以产生第一批 Revision Delta。未来观察名单中的公司提交财报后，实际值
必须按冻结 CIK 和财季匹配，并且只能选择满足 `captured_at < SEC accepted_at` 的最后一条一致
预期。完成这个连接之前，价格和收益窗口继续关闭。

配套的 SEC 监控基线现已覆盖全部 20 家公司，共保存 616 条合格历史 10-Q/10-K，且全部早于
一致预期快照，因此未来 filing 候选数正确地为零。详见
[Forward SEC Filing Monitor v1](forward_filing_monitor_v1.md)。
