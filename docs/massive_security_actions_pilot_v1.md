# Massive 历史证券连续性与公司行为 Pilot v1

## 本轮要解决的问题

28 个事件已经有未调整日线价格和正式交易所日历，但“某个 ticker 在公告日存在”仍不足以证明整段价格窗口属于同一只证券，也不足以直接计算总回报。本轮为每个 `Company × Filing Event × Information Set × Security` 固定五类查询：

1. 价格窗口起点的 `CIK + date + active=true` 证券快照；
2. 价格窗口终点的同口径证券快照；
3. 以 Composite FIGI 查询的 ticker-change 时间线；
4. 窗口内拆股；
5. 窗口内现金分红。

Massive 官方文档说明：[All Tickers](https://massive.com/docs/rest/stocks/tickers/all-tickers) 支持 point-in-time `date`、CIK、active 状态、Share Class FIGI 和 Composite FIGI；[Ticker Events](https://massive.com/docs/rest/stocks/corporate-actions/ticker-events) 可返回 ticker-change 时间线，但该接口仍被官方标为 experimental；[Splits](https://massive.com/docs/rest/stocks/corporate-actions/splits) 和 [Dividends](https://massive.com/docs/rest/stocks/corporate-actions/dividends) 提供拆股生效日、分红除息日和调整因子。四类接口均列为所有 Stocks 套餐可用，Basic 历史窗口为两年。

## 验证口径

只有同时满足以下条件，`historical_symbol_interval_ready` 才为真：

- 起点和终点各有且仅有一条记录，同时严格匹配 CIK、ticker、交易所 MIC、证券类型 `CS`、Share Class FIGI 和 Composite FIGI；
- ticker-event 查询完整，价格窗口内没有 ticker change；
- 三类身份响应没有解析拒绝；
- 所有原始响应先按精确字节保存，并记录 SHA-256。

只有拆股和分红查询均完整、所有记录通过字段与窗口校验、每个生效日都能映射到对应交易所的正式 session open，`corporate_actions_ready` 才为真。零条公司行为是合法结论，但必须来自完整查询；未抓取不能被解释为零。

经过验证的窗口映射写入 `validated_security_mappings.parquet`。拆股和分红被标准化为统一的 `normalized_corporate_actions.parquet`，保留 `security_id`、明确的 `effective_at`、action 类型、现金金额或拆股比例以及原始快照哈希。

## 当前冻结版本

- 目标版本：`massive-security-actions-targets-v1-581121157e9c5ee4`
- 审计版本：`massive-security-actions-audit-v1-36067a60c713841b`
- 事件目标：28
- 每个事件查询：5
- 总计划请求：140
- 原始计划完整：26 / 28；另外 2 个由逐 session 回退完成
- 完整研究证据：28 / 28
- 历史身份区间 ready：28 / 28
- 公司行为窗口 ready：28 / 28
- 原始响应快照：211
- 解析拒绝：0
- 窗口内拆股：0
- 窗口内现金分红：12（11 个 ticker；DX 有 2 条）
- 收益 gate：关闭

原始计划的 `26 / 28` 不是缺失证据：ISPO 和 DNB 的实验性 Ticker Events
Composite FIGI 查询返回 404，随后分别用 37 / 37 和 36 / 36 个正式交易日的 point-in-time
身份快照完成验证。DNB 的补充公司行为查询还增加了一条每股 0.05 美元现金分红。

## 运行

PowerShell 中保留上一轮已加载的 `MASSIVE_API_KEY`，先跑一个目标探针：

```powershell
uv run python -m scripts.build_massive_security_actions_pilot --collect --limit 1
```

单个目标通常发出 5 个请求；按 Basic 的 12.1 秒间隔，约需一分钟。若输出为 `completed=1 failed=0 requests=5 blocked=None`，再抓剩余目标：

```powershell
uv run python -m scripts.build_massive_security_actions_pilot --collect --limit 27
```

完整批次约需 27 分钟。任务可中断后重跑；已经完成的 query 会复用按哈希保存的快照，不重复请求。完成后重建总就绪表：

```powershell
uv run python -m scripts.build_sec_price_readiness
```

### 实验接口返回 404 时

Ticker Events 是实验接口，个别已退市证券的 Composite FIGI 可能返回 404。404 不等于“没有改名”，因此不能被吞掉或当作空事件。收集器会继续补抓该目标的拆股和分红；随后可用官方 point-in-time Ticker 接口，对价格窗口内的每一个正式交易所 session 逐日验证 CIK、ticker、MIC、Share Class FIGI 和 Composite FIGI：

```powershell
uv run python -m scripts.build_massive_security_actions_pilot --collect --collect-session-fallback --limit 2
```

该回退只选择已经明确记录 ticker-event 404 的目标。现有起止点和公司行为快照会复用；按当前两个目标约需新增 79 次以内请求，具体数量由各自正式 session 数决定。只有每个 session 都恰好取得同一证券身份，区间才会以 `massive_pit_every_exchange_session` 方法通过。

## 仍然没有解决的门槛

本轮只验证这 28 个近年事件窗口，不代表 222 个候选事件或总体样本已经完成证券身份映射。Ticker Events 是实验接口，因此其结论会明确保留该标记。更重要的是，Massive 的 inactive/delisted 状态不是退市回报；在退市现金结算、终值或明确缺失原因尚未进入数据合同前，`delisting_outcome_gate` 和真实收益研究仍保持关闭。
