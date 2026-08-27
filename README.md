# Filing-to-Signal Lab

**Bias-Aware Backtesting Framework · 财报量化信号验证框架**

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

Filing-to-Signal Lab investigates a falsifiable question:

> Do markets price a firm's current state, the change in that state, or the innovation—the
> difference between newly disclosed information and the expectation formed immediately before
> the filing? How much do hindsight revisions and historical-data biases distort the answer?

> **Research Preview — 2026-08-27:** the point-in-time data, bias-audit, event-return, and first
> model-ready feature pipelines are operational. The return effectiveness of a standalone
> gross-margin Innovation signal has **not** been established. This repository is an educational
> research system, not investment advice, a stock-selection model, or a validated trading
> strategy.

### Current empirical result

The first return study froze 58 candidate filing events before observing their returns. The final
complete-case sample contains 49 events from 15 issuers, each with Level, Delta, Raw Innovation,
Standardized Innovation, and a 20-trading-day market-excess return.

- Standardized Innovation has a Spearman IC of **+0.045**, with an issuer-clustered 95% interval
  of **[-0.316, +0.356]**. The point estimate is slightly positive, but uncertainty is large.
- Innovation minus Delta has an IC difference of **+0.064** with bootstrap **p = 0.697**; no
  incremental advantage is established.
- The five-day Innovation IC is **+0.204**, but it is a secondary sensitivity result and is not
  significant after Benjamini–Hochberg correction.
- Keeping only companies that survive in hindsight raises the sample's mean market-excess return
  by **1.81 percentage points**.
- Restoring one negative event across the OB→TEAD ticker transition reduces Innovation IC from
  **+0.094** to **+0.045**. The frozen verdict remains
  `effectiveness_not_established`.

The positive result is therefore methodological, not a claim of tradable alpha: the system can
detect and quantify survivorship, security-identity, announcement-timing, and future-data errors
large enough to alter a research conclusion. See the
[longitudinal effectiveness study](docs/longitudinal_effectiveness_v1.md), the
[preregistration](docs/preliminary_effectiveness_preregistration_v1.md), and the
[OB→TEAD repair audit](docs/ticker_transition_repair_v1.md).

The next model round now has a separate point-in-time feature store and a frozen seven-model
baseline tournament. Because all 58 existing event outcomes had already been inspected, they are
permanently marked as training-only rather than reused as validation. Of those events, 35 have a
complete filing-plus-market feature vector. The current tournament correctly emits zero
validation predictions and the verdict `insufficient_unobserved_validation_evidence`; its sealed
holdout begins on 2026-08-28. See the
[Tradeable Dataset and Baseline Tournament v1](docs/tradeable_dataset_and_baseline_tournament_v1.md).

### Research design

The primary representation study compares four views of the same company × filing event ×
information set:

- **Level:** the firm's disclosed financial state.
- **Delta:** the year-over-year change in that state.
- **Raw Innovation:** actual value minus a strictly out-of-sample, pre-event forecast.
- **Standardized Innovation:** Raw Innovation scaled only by strictly earlier out-of-sample
  forecast errors.

All representations are compared on the same complete-case event sample. Forecast cutoffs,
training endpoints, filing acceptance timestamps, price types, and tradable windows are explicit.
Originally reported values power the unbiased signal; later/restated values are used only for
look-ahead-bias experiments.

The application also retains a deterministic simulation that demonstrates how three biases can
make a weak signal look strong:

- **Look-ahead bias:** backfilling later/restated accounting values into historical decisions.
- **Survivorship bias:** deleting companies that later delist from the historical universe.
- **Announcement-timing bias:** assuming execution at the announcement close instead of the
  second trading-day open.

Simulated portfolio performance is a calibrated mechanism demonstration, not market evidence.
The Apple case is real disclosure evidence, but it shows that revisions can change a signal—not
that the signal predicts returns.

The formal protocol is documented in
[State, Change, and Innovation](docs/research_protocol.md).

### Quick start

Python 3.10+ and [uv](https://docs.astral.sh/uv/) are recommended:

```bash
uv sync --locked --dev
uv run python -m scripts.run_public_reproduction
uv run streamlit run app.py
```

The reproduction command needs neither an SEC identity nor a market-data API key. It recomputes
the deterministic simulation and Apple 2009 case, verifies the frozen 49-event aggregate
snapshot, and writes:

```text
output/public_reproduction/public_reproduction_summary.json
```

Because licensed event-level prices are not distributed, this command does not pretend to
recompute the complete empirical return ledger. The evidence levels and release boundary are
defined in the [Public Release Manifest](docs/public_release_manifest.md).

A conventional environment is also supported:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run every public quality gate with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -m scripts.audit_public_release
uv run pytest
```

### Public evidence and data boundary

| Layer | What the public repository can reproduce | Evidence type |
|---|---|---|
| Bias simulation | Recompute all scenarios from a fixed seed | Deterministic simulation |
| Apple 2009 | Re-derive gross margin and signal distortion from a curated SEC slice | Real filing mechanism |
| 49-event study | Verify versioned aggregate statistics and the frozen verdict | Frozen aggregate snapshot |
| Full empirical rebuild | Recollect SEC and market data with the user's own identity and credentials | Credentialed reconstruction |

Git contains code, tests, methods, small SEC example slices, version identifiers, and aggregate
results. It excludes raw SEC responses, processed event panels, Massive/Polygon market payloads,
generated reports, credentials, and temporary renders. In a fresh clone, research views 02–05
fall back to `data/examples/research_preview_summary.json`; when local research assets exist,
the application enables event-level audit views and downloads.

See [Data and artifact policy](DATA_LICENSE.md),
[Public Research Preview checklist](docs/public_release_checklist.md), and
[Public Release Manifest](docs/public_release_manifest.md).

### Project layout

```text
data_generator.py          # Deterministic PIT filings, revisions, delistings, and prices
signals.py                 # SUE and gross-margin change using prior information only
backtest.py                # Event-driven long-short backtest and explicit bias switches
event_panel.py             # Company × Filing Event × Information Set contract
forecast_tournament.py     # M0–M4 strictly PIT forecasting tournament
revision_robustness.py     # Walk-forward revision risk and decision safety margins
revision_tail_audit.py     # Filing-source review of revision-tail events
tradeable_dataset.py       # PIT filing/market features, labels, and evidence roles
baseline_alpha_tournament.py # Same-sample walk-forward baseline models and gates
app.py                     # Streamlit research interface
sec_data/                  # SEC collection, PIT cleaning, identity, and market adapters
scripts/                   # Reproducible research and audit entry points
data/examples/             # Small public SEC slices and frozen aggregate preview
tests/                     # Timing, bias, reproducibility, and acceptance tests
docs/                      # Protocols, results, calibration, and limitations
```

### Research milestones

1. **Apple 2009 look-ahead case.** The original 10-K and a 10-K/A accepted roughly 90 days later
   contain 210 strictly comparable XBRL facts, 77 of which changed. The curated 13-fact slice
   shows 2009 gross margin moving from 35.96% to 40.14%, and its year-over-year signal from
   +1.65 pp to +4.94 pp. Backfilling the amendment inflates the historical signal by 3.29 pp.
   This was retrospective accounting-standard adoption, not evidence of fraud.

2. **Level / Delta / Innovation.** Strict PIT representations were built for an 82-company core
   cohort. The first Innovation specification remained highly correlated with Delta, motivating
   a preregistered forecasting tournament instead of tuning against return outcomes.

3. **Forecast tournament.** M0–M4 were evaluated on 812 common PIT events from 71 companies.
   Company-aware OLS (M2) won with 35.4% MAE skill and improved on the benchmark in all six
   eligible years. This establishes financial forecast skill, not return predictability.

4. **Revision robustness.** Among 503 decision-time events with enough history, the 95% median
   revision-at-risk margin was 0.166 gross-margin percentage points. Empirical coverage reached
   97.4% over 417 later-resolved events, with three realized direction flips. A proposed
   structural-change filter missed those flips and was rejected.

5. **Historical-universe and market-data controls.** The SEC catalog contains 1,057,454 target
   filings and 13,106 candidate CIKs. PIT Assets are matched for 62,963 of 74,410 filing-frame
   issuer-years. A frozen 500-observation probability sample separates representative inference
   from deliberately enriched stress cases. The 28-event price pilot also demonstrates that
   deleting three securities that delisted only after the formal exit raises mean excess return
   by 1.76 pp.

6. **Longitudinal effectiveness study.** After exchange-calendar, corporate-action, delisting,
   benchmark, and ticker-transition controls, 49 complete events from 15 issuers passed the
   preregistered data gate. The final result is
   `effectiveness_not_established`, which is the project's current honest stopping point.

7. **Tradeable Dataset and baseline tournament.** The 58 already-inspected outcomes are frozen as
   training-only. A same-sample filing/market feature store contains 35 complete model rows, but
   no unobserved validation event yet. The seven-model walk-forward tournament therefore stops at
   `insufficient_unobserved_validation_evidence` without opening the sealed holdout.

Detailed milestone documents are linked from [docs/](docs/).

### Backtest conventions

For each simulated reporting-quarter cross-section:

1. Form signals using information available at the announcement.
2. Enter at the second trading-day open after the announcement.
3. Hold for 20 trading days and exit at the close.
4. Buy the strongest 20% and short the weakest 20%.
5. Allocate 50% gross exposure to each side: 100% gross, zero net.
6. Subtract the concurrent simulated benchmark return for IC and quantile analysis.

The application compares “all biases,” “look-ahead corrected,” and “all biases corrected.”
Transparent simulation calibration makes the all-bias annual return approximately 15% and the
fully corrected return below 3%. That gap is a design target, not an estimate of real performance.

### SEC EDGAR workbench

The Streamlit application includes an independent SEC workspace built on the official,
key-free `submissions` and `companyfacts` JSON endpoints. It preserves exact response bytes and
SHA-256 hashes, parses accession and UTC acceptance times, normalizes standard XBRL facts,
strictly pairs amendments, and records rejected observations.

SEC requires automated clients to identify themselves with a real contact address:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@domain.com"
uv run streamlit run app.py
```

The same pipeline can run without the interface:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@domain.com"
uv run python -m sec_data.cli AAPL
uv run python -m sec_data.cli AAPL --include-archives
```

Downloads are stored under `data/sec/` and ignored by Git. This workspace is a disclosure-data
audit, not a complete live backtest; performance research additionally requires PIT security
identity, exchange calendars, corporate actions, prices, and delisting outcomes.

### Full local research pipeline

These commands require the corresponding locally collected data and, where applicable, the
user's own provider credentials:

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
uv run python -m scripts.build_tradeable_dataset
uv run python -m scripts.run_baseline_alpha_tournament
```

Generating the Word research paper also requires local frozen empirical artifacts:

```powershell
uv sync --group reports
uv run python -m scripts.create_academic_effectiveness_paper
```

### Known limitations

- Simulated correlations are calibrated and cannot establish real SUE or gross-margin returns.
- The simulation uses business days rather than a complete exchange calendar and does not model
  holidays, suspensions, price limits, transaction costs, borrow constraints, or market impact.
- Quarterly simulated portfolios rank one reporting cohort at a time even when announcement
  dates differ.
- Simulated delistings do not replicate a complete real-world delisting-return process.
- The current empirical sample is small, issuer-concentrated, and insufficient for a claim of
  stable alpha.
- The first model-ready build has 35 complete training rows and no genuinely unobserved
  validation events; it produces no model-selection or trading claim.
- The application uses deterministic templates and does not require OpenAI.

The software is released under the [MIT License](LICENSE). Third-party data remain subject to
their original terms.

---

<a id="中文"></a>

## 中文

Filing-to-Signal Lab 研究一个可被证伪的问题：

> 市场究竟在定价企业所处的状态、状态的变化，还是新信息——即新披露信息与公告前一刻
> 状态依赖预期之间的差？事后修订数据和历史数据偏差又会怎样扭曲答案？

> **Research Preview — 2026-08-27：** 点时数据、偏差审计、事件收益和第一版训练就绪
> 特征管线已经可以运行，但单一毛利率 Innovation 信号的收益有效性**尚未建立**。
> 本仓库是研究与教育系统，不是投资建议、选股模型或已经验证的交易策略。

### 当前实证结果

第一轮真实收益研究在观察收益前冻结了 58 个候选财报事件。最终共同样本包含 15 家发行人
的 49 个事件，每个事件都同时拥有 Level、Delta、Raw Innovation、Standardized
Innovation 和 20 个交易日市场超额收益。

- 标准化 Innovation 的 Spearman IC 为 **+0.045**，发行人聚类 95% 区间为
  **[-0.316, +0.356]**；方向略正，但不确定性很大。
- Innovation 相对 Delta 的 IC 差为 **+0.064**，bootstrap **p = 0.697**，没有建立增量
  优势。
- 5 日窗口 Innovation IC 为 **+0.204**，但它属于次要敏感性结果，BH 校正后不显著。
- 只保留回看时仍存活的公司，会把样本平均市场超额收益抬高 **1.81 个百分点**。
- 补回一项跨越 OB→TEAD 代码迁移的负向事件后，Innovation IC 从 **+0.094** 降至
  **+0.045**；最终冻结判定仍为 `effectiveness_not_established`。

因此，当前版本的正面成果不是“发现可交易 alpha”，而是系统能够发现并量化足以改变
研究结论的幸存者偏差、证券身份错误、公告时点偏差和未来数据泄漏。完整方法与限制见
[纵向有效性研究](docs/longitudinal_effectiveness_v1.md)、
[预注册协议](docs/preliminary_effectiveness_preregistration_v1.md)和
[OB→TEAD 修复审计](docs/ticker_transition_repair_v1.md)。

下一轮模型已经拥有独立的点时特征库和冻结的七模型基线竞赛。因为现有 58 个事件的收益
都已经被我们看过，它们被永久标记为只可训练，不能再次充当验证证据；其中 35 个事件拥有
完整的财报与市场特征。当前竞赛正确地产生 0 条验证预测，判定为
`insufficient_unobserved_validation_evidence`，封存测试集从 2026-08-28 开始。详见
[Tradeable Dataset and Baseline Tournament v1](docs/tradeable_dataset_and_baseline_tournament_v1.md)。

### 研究设计

核心表示研究在同一个 Company × Filing Event × Information Set 样本上比较四类输入：

- **Level：** 企业当前披露的财务状态。
- **Delta：** 该状态的同比变化。
- **Raw Innovation：** 实际值减去公告前生成的严格样本外预测。
- **Standardized Innovation：** 只使用更早样本外预测误差对 Raw Innovation 标准化。

所有表示必须使用同一个完整事件样本。特征截止时间、训练终点、filing 接受时间、价格
类型和可交易窗口全部显式记录。无偏信号只使用首次披露值；后来修订值只在开启前视偏差
实验时使用。

应用还保留一套确定性模拟，用来展示三种偏差如何把弱信号包装成看似优秀的策略：

- **前视偏差：** 把后来才知道的修订值回填到历史决策时点。
- **幸存者偏差：** 从历史公司世界中删除后来退市的公司。
- **公告时点偏差：** 假设可以在公告日收盘成交，而不是第二个交易日开盘。

模拟组合表现只是一套经过校准的机制演示，不是真实市场证据。Apple 案例是真实披露证据，
但它只证明修订可以改变信号，不证明信号能够预测收益。

正式协议见 [State, Change, and Innovation](docs/research_protocol.md)。

### 快速开始

推荐使用 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --locked --dev
uv run python -m scripts.run_public_reproduction
uv run streamlit run app.py
```

公开复现命令不需要 SEC 身份声明或行情 API Key。它会重新计算确定性模拟和 Apple 2009
案例，校验冻结的 49 事件聚合快照，并写入：

```text
output/public_reproduction/public_reproduction_summary.json
```

由于持牌事件级行情不随 Git 分发，这个命令不会冒充重新计算完整真实收益账本。证据层级
和公开边界见 [Public Release Manifest](docs/public_release_manifest.md)。

也可以使用常规虚拟环境：

```bash
pip install -r requirements.txt
streamlit run app.py
```

运行公开仓库全部质量门：

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run python -m scripts.audit_public_release
uv run pytest
```

### 公开证据与数据边界

| 层级 | 公开仓库可以复现什么 | 证据类型 |
|---|---|---|
| 偏差模拟 | 使用固定种子重新计算所有情景 | 确定性模拟 |
| Apple 2009 | 从精选 SEC 切片重新推导毛利率和信号扭曲 | 真实 filing 机制 |
| 49 事件研究 | 校验带版本的聚合统计和冻结判定 | 冻结聚合快照 |
| 完整实证重建 | 使用者以自己的身份和凭证重新抓取 SEC 与行情 | 有凭证的数据重建 |

Git 只包含代码、测试、方法文档、小型 SEC 示例切片、版本标识和聚合结果。SEC 原始响应、
processed 事件面板、Massive/Polygon 行情、生成报告、凭证和临时渲染不会进入 Git。
在 fresh clone 中，成果 02–05 自动显示
`data/examples/research_preview_summary.json`；本地存在完整研究资产时，应用才会开放
事件级审计页面和下载功能。

详见 [数据与生成物政策](DATA_LICENSE.md)、
[Public Research Preview checklist](docs/public_release_checklist.md)和
[Public Release Manifest](docs/public_release_manifest.md)。

### 项目结构

```text
data_generator.py          # 确定性 PIT 财报、修订、退市和价格模拟
signals.py                 # 只使用历史信息计算 SUE 与毛利率变化
backtest.py                # 事件驱动多空回测和显式偏差开关
event_panel.py             # Company × Filing Event × Information Set 契约
forecast_tournament.py     # M0–M4 严格 PIT 预测竞赛
revision_robustness.py     # 走步修订风险与决策安全边际
revision_tail_audit.py     # 修订尾部事件的 filing 来源复核
tradeable_dataset.py       # PIT 财报/市场特征、收益标签和证据角色
baseline_alpha_tournament.py # 共同样本走步基线模型与研究门槛
app.py                     # Streamlit 研究界面
sec_data/                  # SEC 抓取、PIT 清洗、身份和行情适配器
scripts/                   # 可复现研究与审计入口
data/examples/             # 可公开的小型 SEC 切片与冻结聚合预览
tests/                     # 时序、偏差、复现性和验收测试
docs/                      # 协议、结果、校准与限制
```

### 研究里程碑

1. **Apple 2009 前视偏差案例。** 原始 10-K 与约 90 天后接受的 10-K/A 含有 210 组严格
   可比 XBRL facts，其中 77 组发生变化。精选的 13 条事实显示，2009 毛利率从 35.96%
   变为 40.14%，同比信号从 +1.65 pp 变为 +4.94 pp；回填修订值会把历史信号放大
   3.29 pp。这是收入确认准则的追溯采用，不是欺诈证据。

2. **Level / Delta / Innovation。** 在 82 家核心公司上构建严格 PIT 表示。第一版
   Innovation 与 Delta 仍高度相关，因此项目选择预注册预测竞赛，而不是根据收益结果
   调参。

3. **预测竞赛。** M0–M4 在 71 家公司的 812 个共同 PIT 事件上评估。Company-aware OLS
   （M2）以 35.4% MAE skill 胜出，并在全部六个合格年份优于基准。这建立的是财务预测
   能力，不是收益预测能力。

4. **修订稳健性。** 503 个决策时点事件拥有足够历史，95% 中位 revision-at-risk 安全
   边际为 0.166 个毛利率百分点。在 417 个后来解决的事件上，经验覆盖率为 97.4%，实际
   方向翻转为 3 个。候选结构变化过滤器没有抓到这 3 个翻转，因此被拒绝上线。

5. **历史公司世界与市场数据控制。** SEC catalog 包含 1,057,454 条目标 filing 和
   13,106 个候选 CIK。74,410 个 filing-frame 公司年度中有 62,963 个成功匹配 PIT
   Assets。冻结的 500 条概率样本把代表性推断与刻意富集的压力案例分开。28 事件价格
   pilot 还显示，删除 3 只在正式退出后才退市的证券，会把平均超额收益抬高 1.76 pp。

6. **纵向有效性研究。** 在加入正式交易所日历、公司行动、退市结局、市场基准和 ticker
   迁移控制后，15 家发行人的 49 个完整事件通过预注册数据门。最终结果为
   `effectiveness_not_established`，这是项目目前诚实的停止点。

7. **可交易数据集与基线竞赛。** 58 个已经观察过结果的事件被冻结为只可训练。共同样本
   特征库目前有 35 条完整模型记录，但还没有从未观察过结果的验证事件。因此七模型走步
   竞赛停在 `insufficient_unobserved_validation_evidence`，也没有打开封存测试集。

各阶段详细文件见 [docs/](docs/)。

### 回测约定

每个模拟财报季度形成一个横截面：

1. 用公告时可得的信息形成信号。
2. 在公告后的第二个交易日开盘建仓。
3. 持有 20 个交易日并在收盘退出。
4. 做多信号最高的 20%，做空最低的 20%。
5. 每侧配置 50% 总敞口：总敞口 100%，净敞口为 0。
6. 减去同期模拟基准收益，再计算 IC 和分位数组合。

应用比较“全部偏差”“修正前视偏差”和“修正全部偏差”。透明模拟校准使全部偏差情景的
年化收益约为 15%，全部修正后低于 3%。这个差距是演示目标，不是真实收益估计。

### SEC EDGAR 工作区

Streamlit 应用包含独立的 SEC 工作区，使用官方、无需 API Key 的 `submissions` 和
`companyfacts` JSON 接口。系统保存精确响应字节和 SHA-256，解析 accession 与 UTC
接受时间，标准化 XBRL facts，严格配对 amendment，并记录拒绝观测。

SEC 要求自动客户端使用真实联系地址声明身份：

```powershell
$env:SEC_USER_AGENT="Your Name your.email@domain.com"
uv run streamlit run app.py
```

同一管线可以脱离界面运行：

```powershell
$env:SEC_USER_AGENT="Your Name your.email@domain.com"
uv run python -m sec_data.cli AAPL
uv run python -m sec_data.cli AAPL --include-archives
```

下载内容保存在 `data/sec/` 并被 Git 忽略。这个工作区属于披露数据审计，不是完整实盘
回测；绩效研究还需要 PIT 证券身份、交易所日历、公司行动、行情和退市结局。

### 完整本地研究管线

以下命令需要对应的本地数据；涉及供应商时，需要使用者自己的凭证：

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
uv run python -m scripts.build_tradeable_dataset
uv run python -m scripts.run_baseline_alpha_tournament
```

生成 Word 研究论文还需要本地冻结的实证产物：

```powershell
uv sync --group reports
uv run python -m scripts.create_academic_effectiveness_paper
```

### 已知限制

- 模拟相关性经过人为校准，不能建立真实 SUE 或毛利率信号收益。
- 模拟使用工作日而非完整交易所日历，也没有模拟节假日、停牌、涨跌停、交易成本、
  借券约束和市场冲击。
- 模拟季度组合按单个报告期横截面排名，即使公司公告日期不同。
- 模拟退市没有复刻完整的真实退市收益过程。
- 当前实证样本较小且发行人集中，不足以声称存在稳定 alpha。
- 第一版训练就绪数据只有 35 条完整训练记录，且没有真正未观察的验证事件；当前不能据此
  选择模型或作出交易结论。
- 应用使用确定性文字模板，不依赖 OpenAI。

软件采用 [MIT License](LICENSE)。第三方数据仍受其原始条款约束。
