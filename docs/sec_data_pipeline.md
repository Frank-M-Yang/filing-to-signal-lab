# SEC EDGAR 数据抓取与清洗设计

## 来源与访问规则

管线直接访问 SEC 官方公开接口：

- `https://data.sec.gov/submissions/CIK##########.json`
- `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
- `https://www.sec.gov/files/company_tickers.json`

这些读取接口无需 API Key。客户端强制要求包含真实联系邮箱的 User-Agent，默认每秒
最多 5 次请求，并对 429 和暂时性服务器错误执行有上限的退避重试。SEC 官方上限为
每秒 10 次；本项目选择更保守的速度。

官方参考：

- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data

## 分层

```text
SEC JSON
   -> identified/rate-limited client
   -> immutable SHA-256 raw snapshots
   -> submissions cleaner (accession + accepted_at)
   -> companyfacts cleaner (tag + unit + period + value)
   -> amendment comparator
   -> normalized CSVs + manifest + rejection table
```

原始响应与标准化结果分开保存。标准化数据集版本由所有输入 snapshot hash 决定；相同
输入不会覆盖不同输入形成的数据集。

## 点时规则

`companyfacts.filed` 只有日期，不能建立盘中信息时点。每个 fact 必须用 accession 匹配
`submissions.acceptanceDateTime`。缺少匹配时，该行进入
`missing_acceptance_timestamp` 拒绝记录，不使用 filing date 回填。

## 修订比较

10-K/A 或 10-Q/A 中的事实只与更早的非修正文件比较，并要求以下字段全部一致：

- taxonomy；
- XBRL tag；
- unit；
- period start/end；
- 基础表单类型。

选择修正接受时间之前最近的一条原始披露。新增加但无可比原始值的事实标记为
`no_prior_match`，不会把原始值填成零。

## 当前限制

- Companyfacts 只聚合标准 taxonomy、整家公司层级的 facts；公司扩展标签和分部维度可能
  缺失。
- 当前 API 是 SEC 当前视图，不能单独证明 API 内容本身在历史每一天的状态；长期研究应
  定期归档原始响应或解析 filing-level inline XBRL。
- 10-K/A、10-Q/A 不一定代表数值重述，也可能只修正文案或补充披露；必须查看比较状态和
  原始文件。
- 同一经济概念可能使用不同 tag；当前不做未经版本化的标签替换。
- 10-Q 中可能同时存在单季和年初至今 duration facts；进入信号面板前仍需明确季度选择和
  Q4 推导规则。
- Ticker 映射是当前映射，不是历史证券主表。
- 当前没有真实价格、交易日历、公司行为、停牌或退市收益适配器，因此不会产生真实绩效
  结论。
