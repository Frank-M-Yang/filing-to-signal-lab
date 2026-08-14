# Massive 退市结果与窗口终值 Pilot v1

## 本轮要回答的问题

28 个近年财报事件已经具备未调整日线、正式交易所日历、稳定证券身份，以及窗口内拆股和分红证据。本轮继续回答：这只证券是否在策略的正式退出时点之前停止交易，以及窗口收益是否因此需要额外的退市回报或终值。

这里严格区分两件事：

1. 公司后来退市，不代表本次 20 个交易日的事件窗口受到退市影响；
2. 证券在正式退出日当日或之前停止交易，才需要可核验的退市现金结算、终值或其他回收价值。

Massive 的 `active=false` 和 `delisted_utc` 可以提供停止交易状态及最后交易日，但不是退市回报。因此本项目绝不把缺失的退市回报填成 0。

## 官方语义与查询合同

本层使用 Massive 的 [All Tickers](https://massive.com/docs/rest/stocks/tickers/all-tickers) 接口。官方文档把 `active=false` 定义为已退市证券，并把 `delisted_utc` 定义为该 ticker 的最后交易日期；[Ticker Overview](https://massive.com/docs/rest/stocks/tickers/ticker-overview) 也指向 All Tickers 来查找已退市 ticker。

每个事件都固定同一个结果观察截止日 `2026-08-14`，并按以下顺序查询：

1. 使用 CIK、截止日和 `active=true` 查询；
2. 用事件时已经验证的 Share Class FIGI 与 CIK 精确匹配普通股；
3. 只有该证券未在 active 结果中出现时，才补查 `active=false`；
4. 不使用当前 ticker 回填历史证券身份。

原始 HTTP 响应在解析前按字节保存，并记录 SHA-256。API Key 只进入 `Authorization` header，不写入 URL、日志、Parquet 或 manifest。Basic 探针继续执行至少 12 秒的请求间隔。

## 分类规则

| 截止日状态 | 与正式退出日的关系 | 事件窗口结论 |
|---|---|---|
| `active=true` 且身份唯一 | 截止日仍活跃 | 窗口退市结果就绪；退市回报不适用 |
| `active=false` 且最后交易日晚于退出日 | 后来退市 | 窗口退市结果就绪；退市回报不适用 |
| `active=false` 且最后交易日不晚于退出日 | 可能影响持有窗口 | 必须取得终值/退市回报；gate 保持关闭 |
| 身份冲突、缺少最后交易日或响应未采集 | 无法判定 | gate 保持关闭并记录明确原因 |

这个结果层只验证已经冻结的 28 个事件，不参与样本选择，也不会删除后来退市的公司。

## 当前结果

```text
massive-delisting-targets-v1-38f85d060d4af2e3
massive-delisting-audit-v1-6f2e240cfb707464
sec-price-readiness-v6-9e06c6d4b248e770
sec-universe-design-v6-df32ee21a25f59ca
```

28/28个目标均已采集并通过：25只证券在结果截止日仍活跃；DNB、ISPO和ASNS分别在正式退出159、285和353天后才退市；窗口内退市为0。因此 `outcomes_ready=28`、`eligible=28`，pilot事件收益门槛通过。完整研究门槛仍因历史总体样本尚未冻结而保持关闭。

## 运行

需要复核或在新环境重建时，可先在已经加载 API Key 的同一个PowerShell窗口运行1个目标：

```powershell
uv run python -m scripts.build_massive_delisting_pilot --collect --limit 1
```

如果输出为 `completed=1 failed=0`，继续收集剩余目标：

```powershell
uv run python -m scripts.build_massive_delisting_pilot --collect --limit 27
uv run python -m scripts.build_sec_price_readiness
uv run python -m scripts.build_sec_universe_design
```

收集器支持断点续跑；完成的目标会被跳过。每个仍活跃目标通常需要一次请求，已退市目标通常需要 active 与 inactive 两次请求。

## Gate 的含义

如果 28 个事件全部是“截止日仍活跃”或“退出后才退市”，`pilot_delisting_outcome_gate` 可以通过，且这些单独事件窗口可进入收益构造的下一步。若任何证券在退出日前退市，本层会明确列出需要终值的事件，不会发布带零回报假设的结果。

即使 28/28 通过，`full_return_study_gate` 仍为 false：这批近年可取样本不等于历史总体样本，完整研究还需要冻结历史成分世界及其可复现的抽样/加权设计。
