# 财报量化信号验证框架

**Bias-Aware Backtesting Framework** 研究一个可被证伪的问题：

> 市场究竟在定价企业所处的状态、状态的变化，还是相对于事前预期的新信息？使用事后
> 修订数据又会怎样扭曲这个结论？

> **Research Preview 状态（2026-08-15）：** 偏差审计与点时事件研究管线已经通过，
> 但单一毛利率 Innovation 的收益有效性尚未建立。本项目是研究与教育工具，不是投资
> 建议、选股模型或已经验证的交易策略。

## 当前研究结论

第一轮真实收益研究在观察收益前冻结了 58 个候选财报事件。最终 49 个事件、15 家发行人
同时具备完整 Level、Delta、Raw Innovation、Standardized Innovation 与 20 个交易日
市场超额收益。主要结果为：

- 标准化 Innovation 的 Spearman IC 为 `+0.045`，发行人聚类 95% 区间为
  `[-0.316, +0.356]`；方向略正，但不确定性很大；
- Innovation 相对 Delta 的 IC 差为 `+0.064`，bootstrap `p=0.697`，没有建立增量优势；
- 5 日窗口 Innovation IC 为 `+0.204`，但属于次要敏感性检验，BH 校正后不显著；
- 仅保留回看时仍存活的公司，会把样本平均市场超额收益抬高 `1.81` 个百分点；
- 补回一项跨越 OB→TEAD 代码迁移的负向事件后，Innovation IC 从 `+0.094` 降至
  `+0.045`，最终冻结判定保持 `effectiveness_not_established`。

因此，当前版本的正面成果不是“发现了可交易 alpha”，而是证明系统能够发现并量化足以
改变结论的幸存者、证券身份、公告时点和未来数据偏差。完整方法、冻结判定及限制见
[纵向有效性研究](docs/longitudinal_effectiveness_v1.md)、
[预注册协议](docs/preliminary_effectiveness_preregistration_v1.md)和
[OB→TEAD 修复审计](docs/ticker_transition_repair_v1.md)。

第一项正式研究只使用毛利率，比较 Level、Delta、Raw Innovation 和 Standardized
Innovation，并区分首次价格反应与随后可交易漂移。现有模拟回测是遗留的偏差机制演示，
用于展示三类常见偏差如何把一个弱信号包装成看似优秀的策略：

- 前视偏差：在历史时点使用后来才知道的重述数据；
- 幸存者偏差：从历史股票池中删除最终退市的公司；
- 公告延迟偏差：假设能在公告日收盘成交，而不是第二个交易日开盘成交。

当前应用中的组合表现均来自确定性模拟数据，只用于研究方法展示，不构成真实市场证据或
投资建议。Apple 案例是真实披露证据，但它只证明修订会改变信号，不证明信号能预测收益。

正式研究协议见 [State, Change, and Innovation](docs/research_protocol.md)。

## 运行

使用 Python 3.10+：

```bash
uv sync --dev
uv run python -m scripts.run_public_reproduction
uv run streamlit run app.py
```

第一条命令不需要 SEC 身份声明或行情 API Key：它会重新计算确定性模拟和 Apple 2009
案例，并校验冻结的 49 事件聚合结果，输出到
`output/public_reproduction/public_reproduction_summary.json`。由于第三方事件级行情不随
Git 分发，这一命令不会声称重新计算完整真实收益研究；证据分层和可推送文件范围见
[Public Release Manifest](docs/public_release_manifest.md)。

也可以使用常规虚拟环境：

```bash
pip install -r requirements.txt
streamlit run app.py
```

默认配置使用 50 只模拟股票、36 个季度和固定随机种子。第一次生成后，Streamlit
会缓存数据，因此切换偏差开关和参数时不需要重新模拟价格。

运行公开仓库的质量门：

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -m scripts.audit_public_release
uv run pytest
```

生成 Word 研究报告需要本地冻结的实证产物，这些大文件不随 Git 分发：

```powershell
uv sync --group reports
uv run python -m scripts.create_academic_effectiveness_paper
```

## 项目结构

```text
data_generator.py  # 模拟点时财报、重述、退市与每日价格
signals.py         # SUE 与毛利率同比变化；只使用历史窗口
backtest.py        # 事件驱动多空回测、偏差开关、指标与情景比较
event_panel.py     # Company × Filing Event × Information Set 契约与 OOS 预测
forecast_tournament.py # M0–M4 严格 PIT 预测竞赛
revision_robustness.py # 走步修订风险、安全边际与信号翻转诊断
revision_tail_audit.py # 尾部事件 SEC 来源复核与候选二阶段过滤器诊断
app.py             # Streamlit 交互界面和静态研究总结
sec_data/          # SEC 客户端、批量快照、历史 filing catalog、点时清洗与修订比较
data/examples/     # 可提交、离线可复现的真实 SEC 案例切片
tests/             # 信号时序、偏差语义、可复现性和验收目标测试
docs/              # 方法、模拟校准和已知限制
```

## 公开仓库的数据边界

Git 仓库只包含代码、测试、方法文档、小型 SEC 示例切片、冻结版本标识和汇总结果。SEC
原始响应、本地 processed 数据、Massive 市场数据、生成报告与临时渲染均被排除。完整实证
账本需要使用者以自己的身份声明和供应商凭证重新采集。代码采用
[MIT License](LICENSE)；第三方数据及生成物边界见 [Data and artifact policy](DATA_LICENSE.md)。
首个公开版本的逐项门槛见 [Public Research Preview checklist](docs/public_release_checklist.md)，
可推送/不可推送边界见 [Public Release Manifest](docs/public_release_manifest.md)。

在 fresh clone 中，成果 02–05 自动显示 `data/examples/research_preview_summary.json` 的
冻结聚合快照；如果本地存在完整研究资产，应用会切换到事件级审计页面和下载功能。

## 回测约定

每个财报季度形成一个横截面：

1. 用公告时可得的财报计算信号；
2. 无偏执行在公告后的第二个交易日开盘建仓；
3. 持有 20 个交易日并在收盘退出；
4. 做多信号最高的 20%，做空最低的 20%；
5. 每侧占 50% 总资金，组合总敞口为 100%、净敞口为 0%；
6. 个股收益减去同期模拟市场收益后，用于 IC 和分位数组合比较。

应用固定比较“全部偏差”“修正前视偏差”“修正全部偏差”三种情景，并可叠加用户
当前选择。默认模拟参数经过透明校准，使全部偏差情景的年化收益接近 15%，全部修正后
低于 3%。这个差距是演示设计，不是对真实市场收益的估计。

## 数据源扩展

`data_generator.py` 中的 `ResearchDataSource` 协议定义了数据入口。
`SimulatedDataSource` 是当前实现。未来 SimFin、Sharadar 或 SEC EDGAR 适配器只需返回
相同的 `ResearchData`：点时财报表、每日个股价格和基准价格。真实适配器必须额外解决
公告时间、历史成分股、退市收益、重述版本和公司行为。

## 已知限制

- 模拟相关性是人为设定的，不能用于评估真实 SUE 或毛利率信号。
- 使用工作日而非完整交易所日历；不模拟节假日、停牌或涨跌停。
- 没有交易成本、借券约束和成交量冲击。
- 季度组合按横截面批次结算；公司公告日可不同，但同一季度统一排名。
- 退市通过股票退出样本和压力收益模拟，未复刻真实退市价格流程。
- 当前应用不调用 OpenAI；文字总结使用确定性模板，离线即可运行。

## SEC 真实数据试验台

应用包含一个独立的 SEC EDGAR 工作区。它使用 SEC 官方、无需 API Key 的
`submissions` 和 `companyfacts` JSON 接口，不依赖收费的 `sec-api` 服务。输入 ticker
或 CIK 后，它会：

1. 按 SEC 要求发送带真实联系邮箱的 `User-Agent`，并将速率限制为每秒 5 次以内；
2. 保存响应原始字节及 SHA-256，不覆盖不同版本；
3. 解析 10-K、10-Q、10-K/A 和 10-Q/A 的 accession 与 UTC 接受时间；
4. 展平标准 XBRL facts，保留标签、单位、期间、表单版本和原始路径；
5. 将修正文件与此前同标签、同单位、同期间的原始披露严格配对；
6. 把缺少接受时间、非法单位或无法验证的记录写入拒绝表；
7. 输出 filings、facts、amendment comparisons、rejections 四个 CSV 和 manifest。

先配置身份声明：

```powershell
$env:SEC_USER_AGENT="Your Name your.email@domain.com"
uv run streamlit run app.py
```

也可以直接在 SEC 工作区输入 User-Agent。邮箱不是 API Key，但 SEC 要求自动客户端
提供可联系的身份。下载内容保存在 `data/sec/` 并被 Git 忽略。

同一管线也可以脱离界面运行：

```powershell
$env:SEC_USER_AGENT="Your Name your.email@domain.com"
uv run python -m sec_data.cli AAPL
# 需要更老的 filing history 时：
uv run python -m sec_data.cli AAPL --include-archives
```

这个工作区当前是“披露数据审计”，不是完整真实回测。没有接入点时历史价格、交易所
日历和退市收益之前，SEC facts 不会被直接送入绩效曲线。

## 成果 01：Apple 2009 前视偏差案例

侧边栏的“成果 01 · Apple 修订案例”是项目的第一个真实数据成果，可离线运行。案例来自
Apple 2009 年 10-K 与 90 天后提交的 10-K/A，展示收入确认准则追溯采用如何改变此前
财年的收入、毛利润、净利润、EPS，以及最终进入策略的毛利率同比信号。

核心结果：

- 完整审计中有 210 组严格可比 XBRL facts，其中 77 组数值发生变化；
- 2009 毛利率从原始披露的 35.96% 变为修订后的 40.14%；
- 2009 毛利率同比信号从当时可得的 +1.65 个百分点变为 +4.94 个百分点；
- 若把修订值回填到原始公告日，信号被放大 3.29 个百分点。

案例只包含 13 条与信号直接相关、发生变化的事实，并保留 accession、UTC 接受时间、
单位、期间和完整导出 SHA-256。它是前视偏差的机制验证，不是 Apple 欺诈或财报错误案例，
也不代表全市场重述分布。详见 [Apple 2009 案例方法](docs/apple_2009_case_study.md)。

更详细的定义见 [研究方法](docs/methodology.md) 和
[模拟与限制](docs/simulation_and_limitations.md)，真实数据设计见
[SEC 数据管线](docs/sec_data_pipeline.md)。

2026 Global 500 批量抓取的口径、覆盖率与局限见
[Global 500 → SEC 季度数据样本](docs/global500_sec_cohort.md)。

82 家核心资产的版本、事实选择规则、严格 Level/Delta 表和人工复核边界见
[核心资产清洗说明](docs/core_asset_cleaning.md)。

第二个真实数据成果已经在 82 家核心资产上构建严格 PIT 的 Level、Delta 和 expanding-OOS
Innovation。方法、样本漏斗以及“v1 Innovation 尚未充分区别于 Delta”的诚实结果见
[Level / Delta / Innovation 研究 v1](docs/representation_research_v1.md)。

在接入收益前，v2 只按 PIT 有效性、样本外预测准确率、跨期稳定性和校准选择模型，明确
禁止以降低 Delta–Innovation 相关性为调参目标。冻结的候选模型、MAE 主指标和数据 gate 见
[Forecast tournament v2 预注册](docs/forecast_tournament_v2_preregistration.md)。

M0–M4 已在 812 个共同点时事件上执行。M2 company-aware OLS 以 35.4% 的 MAE skill
胜出，并在六个合格年份全部改善；M3 和 M4 未超过更简单的 M2。该结论仍只属于财务预测，
不属于收益证据。完整结果见
[Forecast tournament v2 results](docs/forecast_tournament_v2_results.md)。

第四个成果把问题推进到公告时点的决策稳健性：只让已经由后续比较披露或 amendment
解决的历史结果进入修订分布，把仍停留在原始 accession 的零变化事件视为右删失。逐份
核对 15 个超出 95% 缓冲的事件后，4 个 XBRL 语义碰撞被纠正，剩余 11 个均为业务范围或
列报口径的历史重绘，没有确认到经济差错。503 个事件在当时拥有足够历史；95% 中位安全
边际仍为 0.166 个毛利率百分点，在 417 个后来可验证事件上的覆盖率提高到 97.4%，方向
翻转从 4 个降为 3 个。候选的事件时点结构变化过滤器没有抓到这 3 个翻转，因此明确不
上线。见 [Revision Robustness v2 Results](docs/revision_robustness_v2_results.md) 和
[Revision Tail Source Audit v1](docs/revision_tail_audit_v1.md)。这仍是财务信号稳定性证据，
不是收益或仓位建议。

第五阶段的历史公司世界扩库已经完成第一轮数据资产。官方 SEC `submissions.zip` 中覆盖
2016–2025 的 1,757 个历史分片已全部解析，完整 catalog 包含 1,057,454 条目标 filing、
13,106 个具有原始定期报告的候选 CIK，以及退市、注销申报、破产、延迟申报和财报不可信
等独立信号。Representative 与刻意富集的 Stress 样本已经分离；后者没有总体权重，不能
估计市场发生率。

官方 `companyfacts.zip` 也已按原始字节和 SHA-256 保存。严格以 accession、报告期和
`accepted_at` 匹配的 `us-gaap:Assets` 覆盖 62,963 / 74,410 个候选公司年度（84.62%）。
1,771 个低于 1 万美元的极小资产值仅被标记复核，没有删除，以免重新制造大公司偏差。
历史身份层也已启动：冻结 74,410 个公司年度的年份均衡采集顺序，保存 1,500 个 filing
index 和其中 500 个完整 cover/XBRL 验证样本。1,500 个 filing-time 公司身份全部验证，
1,481 个取得当时 SIC；高精度旧式封面登记表解析后，321 个取得 subject-CIK 匹配的
ticker/无 ticker 状态，248 个同时取得交易所。首批还发现 144 个 SIC 6189 资产支持证券主体和 68 个
SIC 6770 blank-check 公司，证明“申报 10-K/10-Q”不能直接等同于普通股公司世界。

两阶段概率样本已经冻结：按 10 年 × 四个 PIT 资产分位数加 size-missing 共 50 层，每层
抽取 10 个公司年度，总计 500 个；已知入选概率和权重精确重构 74,410 个 filing-frame
公司年度。所有 500 个样本 cover 已保存，且不允许因非普通股或解析困难而替补。严格证券
标题复核得到 208 个普通股候选、14 个 blank-check 股权候选和 72 个资产支持证券主体；
还有 200 个证券范围未决，因此加权结果仅是诊断，不能当作市场比例。由于历史证券标题、
价格身份、size 缺失机制和极值语义仍未完成，equity analysis sample 仍未冻结，也没有把
SEC 披露事实接入收益回测。见 [Probability sample v1](docs/probability_sample_v1.md)、
[Filing-time identity v1](docs/filing_cover_history_v1.md) 和
[SEC Historical Universe Pipeline](docs/sec_historical_universe_pipeline.md)。行情接入的强制字段、
供应商能力审计和当前阻塞项见 [Point-in-Time Price Readiness v1](docs/price_readiness_v1.md)。

Massive/Polygon 的第一版 PIT 证券主表适配器已经冻结 222 个候选事件，其中 34 个在免费
Basic 计划文档所述的两年历史窗口内。适配器使用 CIK + 公告日期查询、header 鉴权、原始
字节与 SHA-256 保存，并禁止用当前 ticker 回填历史。本机构建时没有配置 API Key，因此结果
诚实记录为 0 个已抓取、0 个最终价格身份，收益 gate 继续关闭。见
[Massive Point-in-Time Reference Pilot v1](docs/massive_reference_pilot_v1.md)。

在该身份审计中唯一通过且具有稳定 FIGI 的 28 个事件现已进入第二阶段：Massive 未调整日线价格适配器。
它显式请求 `adjusted=false`，保存原始响应及 SHA-256，并只生成“观察到的事件窗口候选”；
在历史证券有效区间、正式交易所日历、公司行为和退市结果齐备前不计算或发布真实收益。
当前28个目标已全部抓取，共1,035根通过校验的日线和28个候选窗口，价格解析拒绝为0。
NYSE/Nasdaq官方2024–2025日历与2025年1月9日临时休市公告也已保存并完成哈希审计。
正式日历使6个盘前公告事件的入场和退出日期各提前一个交易日；28/28个正式窗口均有原始价格。
完整方法见 [Official Exchange Calendar v1](docs/exchange_calendar_v1.md)。
方法与运行命令见
[Massive Unadjusted Daily-Price Pilot v1](docs/massive_price_pilot_v1.md)。

复现三个真实数据研究阶段：

```powershell
uv run python -m scripts.run_representation_research
uv run python -m scripts.run_forecast_tournament
uv run python -m scripts.run_revision_robustness
uv run python -m scripts.build_sec_filing_catalog --start-year 2016 --end-year 2025
uv run python -m scripts.build_sec_size_features
uv run python -m scripts.build_sec_filing_cover_history --artifact-policy index-only --limit 1000
uv run python -m scripts.build_sec_probability_sample
uv run python -m scripts.collect_sec_probability_sample
uv run python -m scripts.build_massive_reference_pilot
uv run python -m scripts.build_massive_price_pilot
uv run python -m scripts.build_exchange_calendar
uv run python -m scripts.build_massive_security_actions_pilot
uv run python -m scripts.build_massive_delisting_pilot
uv run python -m scripts.build_sec_price_readiness
uv run python -m scripts.build_event_return_ledger
uv run python -m scripts.build_pilot_filing_features
uv run python -m scripts.build_event_return_diagnostics
uv run python -m scripts.build_longitudinal_expansion
uv run python -m scripts.build_sec_universe_design
```

历史证券连续性与公司行为层现已为 28 个价格窗口冻结起点身份、终点身份、Ticker 变更、拆股和分红共 140 条查询。完整方法、断点续抓命令和当前 gate 状态见
[Massive 历史证券连续性与公司行为 Pilot v1](docs/massive_security_actions_pilot_v1.md)。28 个事件的退市结果层也已完成：25只证券在结果截止日仍活跃，DNB、ISPO和ASNS均在正式退出后才退市，窗口内退市为0；系统禁止把缺失退市回报填成0。运行方法和分类合同见 [Massive 退市结果与窗口终值 Pilot v1](docs/massive_delisting_outcome_pilot_v1.md)。

第一版真实事件收益账本现已生成28/28条公司总收益：第二个交易日未调整开盘买入、持有20个正式交易日、未调整收盘退出，拆股改变持股数，现金分红保留但不再投资。SPY市场基准的未调整价格、拆股和分红也已完成，28/28条市场超额收益可用；行业超额收益继续等待历史行业映射。完整口径见 [真实事件收益账本 v1](docs/event_return_ledger_v1.md)。

基于这份账本，第一版诊断实验已经冻结：全样本平均20日市场超额收益为 -1.04%，9/28个事件跑赢SPY。若故意用未来退市状态删除3只正式退出后后来退市的证券，均值变为 +0.72%，被抬高1.76个百分点；中位数只抬高0.13个百分点，说明效应主要来自负尾部。现在28/28个目标 filing 已按 accession 和 accepted_at 接入同口径 PIT 财务层；覆盖漏斗为 Level 15、Delta 13、严格样本外 Innovation 11。完整共同样本只有11/28，未达到至少20个事件且80%覆盖的预先承诺闸门，因此系统仍不计算 IC，也不对信号有效性下结论。运行 `uv run python -m scripts.build_pilot_filing_features` 和 `uv run python -m scripts.build_event_return_diagnostics` 后，在 Streamlit 选择“成果 05 · 真实事件收益诊断”。方法和限制见 [28事件 PIT 财务特征 v1.1](docs/pilot_filing_features_v1.md) 与 [真实事件收益诊断 v1.1](docs/event_return_diagnostics_v1.md)。

纵向扩样在读取收益前冻结了 58 个候选事件和至少 47 个完整事件的覆盖门。市场批次、
公司行动、退市结局与 OB→TEAD 证券代码迁移修复完成后，最终得到 49 个完整事件、
15 家发行人和 174 个 event-horizon 观察。20 日主检验以及 1 日、5 日次要窗口均按预注册
规则执行；最终判定为 `effectiveness_not_established`。完整数据门、发行人聚类推断、
稳健性诊断和下一步独立扩样要求见
[纵向有效性研究 v1](docs/longitudinal_effectiveness_v1.md)。
