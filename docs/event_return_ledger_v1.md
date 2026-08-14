# 真实事件收益账本 v1

## 本轮成果

本轮把已经通过证券身份、正式交易所日历、未调整价格、公司行为和退市结果五道门槛的28个财报事件，转换为逐事件、逐现金流可追溯的真实收益账本。

当前离线资产：

```text
massive-benchmark-targets-v1-a960a35ea7310529
massive-benchmark-audit-v1-019053dc665f984e
event-return-ledger-v1-7b824e392f85232e
```

当前结果是 `issuer=28/28`、`market_excess=28/28`、`sector_excess=0/28`。公司收益和同期SPY市场超额收益已经全部生成；行业超额收益等待历史时点行业映射。

## 冻结的收益合同

每个事件采用相同的无偏执行规则：

1. SEC公告后的第二个正式交易日，以常规交易时段未调整开盘价买入一股；
2. 持有20个正式交易日，以未调整收盘价退出；
3. 窗口内拆股按生效日开盘调整持股数；
4. 窗口内现金分红按除息日确认权利，现金保留但不再投资；
5. 除息日恰好等于入场日的分红不计入，因为在除息日开盘买入已无权取得该次分红；
6. 若将来出现退出前退市，必须有终值或退市回报，绝不填0。

一股初始持仓的总收益定义为：

```text
terminal wealth = exit shares × unadjusted exit close + entitled cash dividends
total return    = terminal wealth / unadjusted entry open - 1
```

账本同时保存：拆股调整后的价格收益、现金分红收益、总收益、每条现金流、每个价格/日历/公司行为/退市证据的SHA-256。没有使用供应商的黑箱调整价格。

## 当前28个事件的核对结果

- 28个事件全部得到有限、非缺失的公司总收益；
- 8笔现金分红实际落在精确持有窗口内；
- 4笔已抓分红位于入场前或退出后，因此没有错误计入；
- 窗口内拆股为0；
- 3只后来退市证券仍留在账本中；
- 所有证据哈希均通过格式和关联核对；
- 没有计算年化收益、夏普或信号显著性。

样本公司总收益均值约为-5.13%，中位数约为-5.85%，范围约为-34.41%至+49.44%。这只是一个集中在近年、小规模、可取数事件上的管道核对，既不是市场平均收益，也不能证明任何财报信号有效或无效。

## 市场基准

v1把SPY冻结为明确的标普500总收益代理。基准层独立抓取：

- `adjusted=false` 的SPY日线；
- 查询区间内SPY拆股；
- 查询区间内SPY现金分红。

SPY查询范围为 `2024-11-08` 至 `2025-09-15`。它不是用今天的公司ticker映射历史证券，而是研究合同中明确版本化的基准工具。

运行：

```powershell
uv run python -m scripts.build_event_return_ledger --collect-benchmark
```

命令包含3类可续抓查询，按Basic限速通常约半分钟完成，并会自动重建基准审计和事件收益账本。成功时应看到：

```text
BENCHMARK COLLECT DONE ... completed=3 failed=0 ...
BENCHMARK AUDIT DONE ... ready=True ...
EVENT RETURN LEDGER DONE ... issuer=28/28 market_excess=28/28 ...
```

市场超额收益同时输出两种透明口径：

```text
arithmetic excess = issuer total return - SPY total return
relative wealth   = (1 + issuer return) / (1 + SPY return) - 1
```

## 输出

- `event_return_ledger.parquet`：一行一个财报事件；
- `event_return_cashflows.parquet`：入场、分红、拆股和退出现金流；
- `event_return_evidence.parquet`：每个事件所依赖的原始快照哈希；
- `event_return_contract.json`：冻结的执行和现金流口径；
- `event_return_report.json`：局部门槛与完整研究门槛。

## 仍然关闭的门槛

即使SPY完成并产生28条市场超额收益，`full_return_study_gate` 仍保持false。原因包括：历史成分世界尚未冻结、222个映射候选尚未全部进入价格层、500个抽样单位仍有范围裁决未完成，并且28个事件不足以进行可靠的Level/Delta/Innovation统计检验。

行业基准也不会用今天的GICS或静态行业分类回填。下一阶段需要构建公告时点可得的历史行业映射，再选择明确版本化的行业指数或ETF代理。
