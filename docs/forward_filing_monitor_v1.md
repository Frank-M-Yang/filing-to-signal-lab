# Forward SEC Filing Monitor v1

## English

### Purpose

The forward consensus ledger needs an equally point-in-time actual-value clock. This monitor
freezes the SEC submissions state for the same 20 issuers and later identifies a new 10-Q or 10-K
only after the filing has an SEC acceptance timestamp.

A filing can become a future-event candidate only when:

1. its form is 10-Q, 10-Q/A, 10-K, or 10-K/A;
2. `accepted_at` is strictly later than the consensus `captured_at`;
3. the SEC report date matches the provider target period exactly or lies within a provisional
   seven-day fiscal-calendar tolerance; and
4. the submissions response was observed no earlier than the filing acceptance time.

The seven-day rule does not authorize an actual value. A non-exact date still requires accession-
level XBRL period confirmation.

### Baseline capture

The first identified SEC collection completed 20/20 issuers with zero request failures. Exact
response bytes, SHA-256 hashes, fetch timestamps, response metadata, and the frozen target identity
were preserved. The contact email used in the User-Agent is not written to any artifact.

The baseline contains 616 eligible historical 10-Q/10-K observations. All 616 were accepted before
their corresponding consensus capture, so the correct future-candidate count is zero. An explicit
latest-prior baseline table contains one filing for every issuer:

- latest prior filings: 20/20 issuers;
- closest prior filing: Walmart 10-Q, about 0.37 day before the consensus capture;
- median consensus-to-prior-filing gap: about 28.88 days;
- maximum gap: about 86.18 days; and
- latest baseline acceptance: 2026-08-28 16:11:28 UTC.

This is the intended initial state: the monitor proves what filings already existed when the
consensus snapshot was taken. A later poll can therefore distinguish genuinely new filings from
the historical inventory.

### Current gates

| Gate | Result | Meaning |
|---|---:|---|
| SEC submissions baseline | Pass, 20/20 | Every watched issuer has an immutable baseline |
| Post-consensus target filing | 0 | No target 10-Q/10-K has arrived yet |
| Accession-level XBRL actual | Blocked | There is no eligible new accession to parse |
| Innovation | Blocked | Actual revenue minus eligible consensus cannot yet be computed |
| Return research | Closed | No price outcome is loaded |

Zero candidates is not a negative model result. It is a time-state result: the future accounting
events have not occurred.

### Reproduction

With a compliant `SEC_USER_AGENT` loaded only in the current process:

```powershell
uv run python -m scripts.build_forward_filing_monitor --collect --limit 20
```

Re-running on the same UTC date reuses completed targets. A later date creates a new immutable
snapshot. When a candidate appears, the next stage will collect accession-level XBRL facts and
verify the fiscal period before calculating Innovation.

---

## 中文

### 目的

前向一致预期账本还需要一个同样严格点时的实际值时钟。本监控器为同一批 20 家公司冻结 SEC
submissions 状态；之后只有当新的 10-Q/10-K 获得 SEC 接受时间，才会被识别出来。

一份 filing 必须同时满足以下条件，才能成为未来事件候选：

1. 表单是 10-Q、10-Q/A、10-K 或 10-K/A；
2. `accepted_at` 严格晚于一致预期的 `captured_at`；
3. SEC report date 与供应商目标财季完全一致，或暂时落在七天财务日历容差内；
4. submissions 响应的观察时间不早于 filing 接受时间。

七天容差不能直接授权实际值；日期不完全一致时，仍必须用 accession 级 XBRL 确认财季。

### 首次基线采集

首次合规 SEC 采集完成 20/20 家，零请求失败。原始响应字节、SHA-256、抓取时间、响应元数据和
冻结目标身份均已保存。User-Agent 使用的联系邮箱不会写入任何数据产物。

基线包含 616 条历史 10-Q/10-K 观察；616 条全部早于对应的一致预期抓取时间，因此未来候选数
正确地为零。独立的“最近一份基线财报”表对每家公司保留一行：

- 最近基线 filing 覆盖 20/20 家；
- 距离最近的是 Walmart 10-Q，在共识快照前约 0.37 天；
- 共识抓取与上一份财报之间的中位间隔约 28.88 天；
- 最大间隔约 86.18 天；
- 基线中最晚的接受时间为 2026-08-28 16:11:28 UTC。

这正是我们希望看到的初始状态：系统证明了抓取一致预期时哪些财报已经存在。以后再次轮询时，
就能把真正新增的 filing 与历史库存区分开。

### 当前 Gate

| Gate | 结果 | 含义 |
|---|---:|---|
| SEC submissions 基线 | 通过，20/20 | 每家公司都有不可变基线 |
| 共识之后的目标财报 | 0 | 新的目标 10-Q/10-K 尚未到来 |
| Accession 级 XBRL 实际值 | 阻止 | 没有可解析的新 accession |
| Innovation | 阻止 | 尚不能计算实际收入减合格共识 |
| 收益研究 | 关闭 | 没有加载价格结果 |

零候选不是模型的负面结果，而是时间状态：未来会计事件还没有发生。

### 运行方式

只在当前进程加载合规 `SEC_USER_AGENT` 后运行：

```powershell
uv run python -m scripts.build_forward_filing_monitor --collect --limit 20
```

同一 UTC 日期内重复运行会复用已完成目标；以后日期会生成新的不可变快照。候选出现后，下一阶段
将抓取 accession 级 XBRL facts，确认财季后再计算 Innovation。
