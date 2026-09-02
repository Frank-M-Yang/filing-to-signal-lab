# Shadow market evidence and portfolio allocator v1

## English

### Why this layer exists

A signal intention is not yet a tradable order. Before allocating paper cash,
the system must prove which security represented the issuer at the filing time,
which official sessions define entry and exit, which prices were observable
before sizing, and whether transaction-cost and corporate-action evidence is
complete. Missing evidence blocks the event; it is never filled with a current
ticker, a future price, or a default cost.

### Point-in-time market evidence plan

All 37 prospective events map to one of 20 frozen registration-time security
candidates: eight Nasdaq and twelve NYSE issuers. This is only a candidate
mapping. After a target filing receives an SEC acceptance timestamp, the daily
pipeline creates five immutable query targets:

1. ticker/security identity as of the acceptance date;
2. unadjusted daily bars for the strictly pre-acceptance cost window;
3. unadjusted bars for the planned execution and holding window;
4. split actions; and
5. cash dividends.

Every query has `query_not_before >= sec_accepted_at`. At present no registered
target filing has been accepted, so zero market queries is the correct result.
The frozen registry contains fiscal periods through January 2027, but this does
not make 2027 market data a current input. Calendar loading is event-scoped: it
starts only after an SEC acceptance timestamp exists and loads the 45-day range
around that acceptance. A 2026 event therefore uses only preserved 2026 source
documents. If a future accepted event's execution window crosses into 2027,
that event alone remains blocked until the official 2027 NYSE/Nasdaq documents
are preserved; the existing 2026 calendar is never extrapolated.

The collector/audit is now implemented. It preserves exact Massive response
bytes under their SHA-256 hashes, keeps API credentials out of URLs and
artifacts, and parses the five evidence classes separately. Promotion to an
order candidate requires all of the following to have been observed before the
second official session opens:

- acceptance-date ticker and exchange identity match;
- exactly 20 completed pre-acceptance sessions for the frozen
  overnight-adjusted Corwin-Schultz plus 5-bps cost proxy, clipped to 10–100
  bps;
- a sizing close observed only after the first post-acceptance session closes;
- completed split and dividend queries; and
- a filing observation timestamp no earlier than SEC acceptance.

The collector can append daily execution/action snapshots, but a late snapshot
cannot repair a missed pre-entry clock. No return is calculated by this layer.

### Cash-constrained paper allocation

The paper-account convention is frozen at an initial USD 100,000 NAV. It uses
whole shares, no shorting, and no leverage. For a fully evidenced buy intention:

- target notional is 2% of initial paper NAV;
- quantity is sized from the last eligible pre-entry reference price, never the
  unknown future opening fill;
- half of the estimated round-trip cost is reserved at entry;
- aggregate 20% gross, 2% issuer, ten-position, and available-cash limits are
  enforced in stable event order.

Passing allocation creates `shadow_order_candidates.parquet`, not an order.
Broker submission and live authorization remain false. V1 starts without open
positions; a negative signal therefore means exit-or-avoid and cannot open a
short.

### Current result

- security candidates: 37/37 events, 20 issuers;
- accepted target filings: 0;
- market queries: 0;
- shadow intentions: 0;
- order candidates: 0;
- paper cash available: USD 100,000;
- positions, P&L, and returns: zero/not loaded.

The empty result is evidence that the clocks are enforced, not evidence for or
against signal profitability.

## 中文

### 为什么需要这一层

信号意图不等于可交易订单。在分配模拟资金前，系统必须证明财报时点对应的证券身份、官方入场和
退出交易日、仓位计算前可见的价格，以及成本和公司行为证据是否完整。证据缺失时直接阻塞，绝不
用当前 ticker、未来价格或默认成本补齐。

37 个前瞻事件目前都能映射到 20 个“注册时证券候选”（8 家 Nasdaq、12 家 NYSE），但这不是
未来财报时点已经验证的身份。目标财报出现 SEC acceptance 后，流水线才会生成五类查询：时点
证券身份、公告前成本行情、执行窗口行情、拆股和现金分红。每项查询的最早时间都不得早于 SEC
acceptance。当前没有目标财报被接受，因此查询为零是正确结果。

冻结清单的目标财季延伸到 2027 年 1 月，但这不代表现在把 2027 市场数据当作输入。官方日历只在
某个事件获得 SEC acceptance 后，按该事件前后 45 天动态加载；2026 年内完成执行窗口的事件只
使用已保存的 2026 官方文件。未来若某个事件的执行窗口跨入 2027 年，只有该事件会因为缺少
2027 官方日历而被阻塞，其他 2026 事件不受影响，也绝不会从 2026 日历外推未来交易日。

采集与审计现已实现：所有 Massive 原始响应先按 SHA-256 不可变保存，密钥不会进入 URL、日志或
产物。只有在第二个正式交易日开盘前完成 acceptance 时点证券身份匹配、严格公告前 20 个交易日
成本估计、首个公告后交易日收盘后取得的仓位参考价，以及拆股和分红查询，市场证据才会通过。
迟到的数据不能倒填错过的入场时钟；这一层也不计算任何收益。

模拟账户初始净值冻结为 100,000 美元，只允许整数股、现金多头、无卖空、无杠杆。完全通过证据
Gate 的买入意图以 2% 模拟净值为目标，用入场前可见参考价格计算股数，并预留一半估算往返成本；
同时检查 20% 总投入、2% 单发行人、最多十个仓位和可用现金限制。通过后产生的仍是“影子订单
候选”，不会连接券商或授权实盘。

当前 37 个事件均未出现目标财报或 Innovation，因此账户保持 100,000 美元全现金，意图、订单
候选、仓位和收益均为零。这说明时间边界正常工作，并不代表信号已经有效或无效。
