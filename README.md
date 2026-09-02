# Filing-to-Signal Lab

**Bias-Aware Backtesting Framework · 财报量化信号验证框架**

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

Filing-to-Signal Lab investigates a falsifiable question:

> Do markets price a firm's current state, the change in that state, or the innovation—the
> difference between newly disclosed information and the expectation formed immediately before
> the filing? How much do hindsight revisions and historical-data biases distort the answer?

> **Research Preview — 2026-08-28:** the point-in-time data, bias-audit, event-return, and first
> model-ready feature pipelines are operational. The return effectiveness of a standalone
> gross-margin Innovation signal has **not** been established. This repository is an educational
> research system, not investment advice, a stock-selection model, or a validated trading
> strategy.

### Current empirical result

A new external-expectation experiment provides the project's first strong positive forecast
result. On 35 exact-quarter revenue-guidance events from 13 issuers, prior management guidance
achieves **2.81% MAPE**, versus **13.73%** for a strictly point-in-time prior-year same-quarter
baseline—a **79.53% relative improvement**. The event win rate is 77.14%, the issuer-clustered
APE-difference interval is fully below zero, and the improvement remains at least 76.11% after
leaving out any one issuer. This is conditional forecast evidence, not alpha: exact comparable
guidance covers only 9.41% of the 372 candidate events. The outcome-blind expansion now provides
23 authorized returns from eight issuers; their descriptive rank correlation is -0.0069, but the
formal return gate is withheld because issuer breadth and the three-event negative tail are
insufficient. See [Point-in-Time Management Guidance as an Expectation Model](docs/revenue_guidance_innovation_v1.md)
and the [first return diagnostic](docs/revenue_guidance_return_diagnostic_v1.md).

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

The first unseen-sample expansion now freezes 2,503 candidate events from 180 issuers without
loading returns. SEC reconstruction produces 1,074 complete filing-feature events from 87
unobserved issuers; only 319 also fall inside the currently documented Basic price-history
window. The bottleneck is therefore measurable: both issuer breadth and longer price history are
still required. See [Unseen Validation Expansion v1](docs/validation_expansion_v1.md).

A separate forward-only consensus ledger began on 2026-08-29. Its first immutable capture covers
20/20 core issuers and 37 future quarterly revenue estimates. The SEC monitor covers the same 20
issuers, but no target-period filing has yet been accepted; therefore Revision Delta, quarterly
actuals, and Innovation correctly remain empty. The pipeline now supports exact-accession direct
quarterly facts and strict 10-K Q4 derivation as FY minus an earlier original 10-Q nine-month fact.
Run `uv run python -m scripts.run_forward_research_daily` to rebuild the credential-free local
state, or see [the daily pipeline contract](docs/forward_daily_pipeline_v1.md).

The daily pipeline now also emits an immutable prospective event state machine. It records one
current state and a two-clock transition trail for every frozen event, names the reason an event is
waiting or rejected, and fails if an event moves backwards across daily snapshots. The current
37/37 events are `consensus_revision_ready`; no paper or live position is authorized. See
[Prospective Event State Machine v1](docs/prospective_event_state_machine_v1.md).

An outcome-free consensus-quality layer now separates numeric revisions from provider-structure
changes. Of 37 frozen events, 33 are currently clean, two MCK periods require review, and the NVDA
and WMT 2026-10-31 periods are quarantined. All 37 remain observable, but only the 33 clean events
can progress into downstream Innovation or paper-decision states. See
[Prospective Consensus Quality v1](docs/prospective_consensus_quality_v1.md).

The first cash-only shadow policy is now preregistered on the same 37 still-unobserved events. A
Raw Revenue Innovation of at least +1% maps to a 2%-weight buy intention, at most -1% maps to exit
or cash, and the middle band does not trade. Gross invested weight is capped at 20%, shorting and
leverage are disabled, and the v3 timing and cost rules are retained. The policy-aware ledger still
creates no order until calendar, price, cost, cash, and portfolio evidence pass. See
[Shadow Trading Policy v1](docs/shadow_trading_policy_v1.md) and
[Policy-Aware Shadow Order Ledger v2](docs/shadow_order_ledger_v2.md).

A point-in-time market-evidence collector and cash-constrained paper allocator now sit after that
ledger. The 37 events map to 20 registration-time security candidates, but acceptance-date identity,
official calendar, unadjusted price, cost, split, and dividend evidence must all be observed before
the second official session opens. Exact provider bytes are preserved under SHA-256 before parsing.
The paper account starts at USD 100,000, uses whole shares, and retains
the 20% gross/no-short/no-leverage limits. No target filing has yet been accepted, so queries,
intentions, candidates, positions, and returns correctly remain zero. See
[Shadow Market Evidence and Portfolio Allocator v1](docs/shadow_market_and_portfolio_v1.md).

Market-Innovation v3 is now preregistered on that untouched forward ledger. It freezes 37 future
company-quarter events from 20 issuers across six period-ends, with no actual or return value
loaded. The primary score is reported revenue relative to the last analyst revenue consensus
observable before SEC acceptance; Reaction is diagnostic rather than part of the score. The 16
failed Conditional Drift holdout events are permanently firewalled, and both O1 and O3 are retired.
Formal return evaluation requires at least 30 complete events, 15 issuers, and eight Innovations on
each side of zero. A registry-aware readiness ledger now covers 37/37 events and 20 SEC issuers;
it continues consensus capture after fiscal period-end and stops only at each event's SEC
acceptance. See [Prospective Market-Innovation v3](docs/prospective_market_innovation_v3.md).

Conditional Drift Study v1 now joins State, Delta, Innovation, and the market reaction observable
before the formal second-session-open entry on the same events. It produces gross and explicit
10/25/50-bps cost-adjusted 5-, 20-, and 60-session labels. Thirty-five events from 15 issuers have
the complete same-event vector; only 17 and 14 events receive point-in-time development forecasts
at five and 20 sessions because training labels must mature before each prediction. Reaction and
Innovation rank the later 20-session outcomes differently, but all events are historical
training-only and the unobserved-validation gate remains closed. These are development diagnostics,
not alpha evidence. See [Conditional Drift Study v1](docs/conditional_drift_study_v1.md).

The first optimization round rejects a broad feature expansion. Decomposing Reaction and adding
every v2 feature reduces the 20-session development IC by 0.163 and worsens MAE. A much narrower
Innovation–Reaction alignment block improves rank correlation by 0.027 at five sessions and 0.035
at 20 sessions; its 20-session leave-one-issuer-out IC remains positive, but only 14 development
predictions exist. Alignment-only is therefore frozen as a low-confidence candidate for future
external validation, while the full v2 specification is discarded. See
[Conditional Drift Optimization v2](docs/conditional_drift_optimization_v2.md).

That candidate is now operationally frozen. The model bundle records the exact feature order,
scaler parameters, Ridge coefficients, intercept, and final training-label cutoff. Of 63
post-cutoff complete filing events, 47 are quarantined because outcomes already exist locally.
The remaining 16 were scored from outcome-free market and pre-entry Reaction features; no holdout
metric or P&L has been computed. Only 7/16 lie fully inside the development feature range and two
are severe extrapolations, so the covariate-overlap, external-validation, and live-trading gates
remain closed. See the
[Conditional Drift Forward Candidate v1](docs/conditional_drift_forward_candidate_v1.md).

The preregistered temporal evaluation is now complete, and the candidate did not validate. On the
exact 16 pre-scored events, Alignment-only has a 20-session Spearman IC of **-0.106**, versus
**-0.074** for the challenger; the incremental IC is **-0.032**, with issuer-clustered 95%
interval **[-0.367, +0.212]**. Its top-minus-bottom tertile spread is **-13.50 percentage points**,
and only 1/7 frozen verdict checks passes. Endpoint review confirms that the two >50% tail returns
are continuous 20-session, non-OTC price paths without split/dividend events. The verdict is
`candidate_effectiveness_not_established`; these 16 outcomes are now permanently observed and
will not be reused as validation. See the
[Conditional Drift Temporal Holdout Evaluation v1](docs/conditional_drift_holdout_evaluation_v1.md).

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
sec_data/                  # SEC collection, PIT cleaning, validation expansion, and market adapters
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

8. **Unseen validation expansion.** An outcome-blind 2022–2025 registry contains 2,503 new
   candidate events. SEC data alone yields 1,074 complete events from 87 unobserved issuers. The
   current Basic price window retains only 319, so the project expands the frozen SEC sample and
   seeks longer price history instead of weakening the model gates.

9. **Management-guidance expectation.** On 35 exact-quarter revenue-guidance events, prior
   guidance reduces MAPE from 13.73% to 2.81% relative to the PIT seasonal baseline. All forecast
   gates pass, including issuer-cluster and leave-one-issuer-out robustness. The separate return
   gate remains closed at 12 events / 7 issuers, so no IC or alpha claim is made.

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
uv run python -m scripts.build_validation_expansion
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
- The first unseen expansion has sufficient planned events but only 87 filing-feature-complete
  issuers, while the current Basic price window covers only six quarters.
- The application uses deterministic templates and does not require OpenAI.

The software is released under the [MIT License](LICENSE). Third-party data remain subject to
their original terms.

---

<a id="中文"></a>

## 中文

Filing-to-Signal Lab 研究一个可被证伪的问题：

> 市场究竟在定价企业所处的状态、状态的变化，还是新信息——即新披露信息与公告前一刻
> 状态依赖预期之间的差？事后修订数据和历史数据偏差又会怎样扭曲答案？

> **Research Preview — 2026-08-28：** 点时数据、偏差审计、事件收益和第一版训练就绪
> 特征管线已经可以运行，但单一毛利率 Innovation 信号的收益有效性**尚未建立**。
> 本仓库是研究与教育系统，不是投资建议、选股模型或已经验证的交易策略。

### 当前实证结果

新的外部预期实验给出了项目目前第一个较强的正面预测结果。在 13 家发行人的 35 个精确同季度
收入指引事件上，既往管理层指引的 MAPE 为 **2.81%**，严格点时的去年同季度基准为
**13.73%**，相对改善 **79.53%**。逐事件胜率为 77.14%，发行人聚类的 APE 差区间完全低于
零；依次剔除任何一家发行人后，改善仍至少为 76.11%。但这只是有条件的预测证据，不是
alpha：严格可比指引只覆盖 372 个候选事件的 9.41%。无结果选样的扩展现已得到 8 家发行人的
23 个授权收益，描述性秩相关为 -0.0069；但发行人广度和仅 3 个事件的负向尾部不足，因此正式
收益 Gate 仍未打开。详见
[Point-in-Time Management Guidance as an Expectation Model](docs/revenue_guidance_innovation_v1.md)
和[第一轮收益诊断](docs/revenue_guidance_return_diagnostic_v1.md)。

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

第一轮未观察样本扩展在完全不加载收益的情况下，冻结了 180 家发行人的 2,503 个候选事件。
SEC 重建后有 87 家未观察发行人的 1,074 个事件具备完整财报特征，但其中只有 319 个同时
落入当前记录的 Basic 行情历史窗口。瓶颈因此已经量化：仍需要扩大公司覆盖并取得更长的
历史行情。详见 [Unseen Validation Expansion v1](docs/validation_expansion_v1.md)。

另一条完全前向的一致预期账本从 2026-08-29 开始。第一次不可变快照覆盖 20/20 家核心公司
和 37 条未来季度收入预期；SEC 观察层也覆盖同样 20 家公司，但目标期间财报尚未被 SEC
接受，因此 Revision Delta、季度实际值和 Innovation 正确地保持为空。流水线现已支持精确
accession 的直接季度事实，以及用全年值减去此前原始 10-Q 九个月累计值来严格推导 10-K 的
Q4。运行 `uv run python -m scripts.run_forward_research_daily` 可在不联网的情况下重建当前
本地状态，方法见[日常流水线协议](docs/forward_daily_pipeline_v1.md)。

每日流水线现在还会生成不可变的前瞻事件状态机：它为每个冻结事件保存一个当前状态和同时包含
生效时点、实际观察时点的转换轨迹，明确记录等待或拒绝原因，并在事件跨日倒退时直接失败。
当前 37/37 个事件均处于 `consensus_revision_ready`，状态机没有授权任何模拟或真实仓位。详见
[Prospective Event State Machine v1](docs/prospective_event_state_machine_v1.md)。

一层不读取收益的一致预期质量审计现已把数值修订与供应商结构变化分开。37 个冻结事件中，当前
33 个干净、MCK 两个财季需要复核、NVDA 与 WMT 的 2026-10-31 财季被隔离。全部事件仍会继续
观察，但只有 33 个干净事件能够进入后续 Innovation 或模拟决策状态。详见
[Prospective Consensus Quality v1](docs/prospective_consensus_quality_v1.md)。

第一版仅现金影子交易政策已经在同一批 37 个尚未观察结果的事件上预注册：Raw Revenue
Innovation 至少 +1% 时生成 2% 权重的买入意图，至多 -1% 时退出或保持现金，中间区间不交易；
总投入限制为 20%，禁止卖空和杠杆，并沿用 v3 的时间与成本规则。政策感知账本在日历、价格、
成本、现金和组合证据通过前仍不会创建订单。详见
[Shadow Trading Policy v1](docs/shadow_trading_policy_v1.md)和
[Policy-Aware Shadow Order Ledger v2](docs/shadow_order_ledger_v2.md)。

政策账本之后现在增加了点时市场证据采集审计和现金约束模拟组合分配器。37 个事件能映射到 20 个
注册时证券候选，但在第二个正式交易日开盘前，必须重新验证 acceptance 时点身份、官方交易日历、
未复权价格、成本、拆股和分红证据；供应商原始响应在解析前按 SHA-256 不可变保存。模拟账户以
100,000 美元起始、只使用整数股，并继续执行总投入 20%、
禁止卖空和杠杆的限制。当前没有目标财报被 SEC 接受，因此查询、意图、订单候选、仓位和收益均
保持为零。详见
[Shadow Market Evidence and Portfolio Allocator v1](docs/shadow_market_and_portfolio_v1.md)。

Market-Innovation v3 已经在这套未观察前向账本上完成预注册：20 家发行人的 37 个未来公司—财季
事件、6 个目标期末已经冻结，注册时没有加载任何实际值或收益。主信号固定为“实际收入相对于
SEC acceptance 前最后一次可验证分析师收入一致预期的偏离”；Reaction 只作诊断，不进入主
信号。失败的 16 个 Conditional Drift 留出事件被永久隔离，O1 与 O3 均已退休。正式收益检验
至少要求 30 个完整事件、15 家发行人，且正负 Innovation 各不少于 8 个。详见
[Prospective Market-Innovation v3](docs/prospective_market_innovation_v3.md)。注册表驱动的就绪账本
现已覆盖 37/37 个事件和 20 家 SEC 发行人；一致预期采集不会在财季末停止，而会持续到每个
事件各自的 SEC acceptance。

Conditional Drift Study v1 现已把 State、Delta、Innovation 和正式第二交易日开盘入场前
能够观察到的市场 Reaction 合并到同一事件，并生成 5 / 20 / 60 日毛收益和显式扣除
10 / 25 / 50 bps 往返成本后的标签。15 家发行人的 35 个事件拥有完整同事件输入；由于每次
预测前训练标签必须已经成熟，5 日和 20 日分别只有 17 和 14 个事件得到点时开发预测。
Reaction 与 Innovation 对后续 20 日结果给出了不同排序，但所有事件仍是 historical
training-only，未见样本验证闸门继续关闭，因此这只是开发诊断，不是 alpha 证据。详见
[Conditional Drift Study v1](docs/conditional_drift_study_v1.md)。

第一轮优化明确淘汰了“多堆特征”的方案：拆分 Reaction 并加入全部 v2 特征后，20 日开发期
IC 相对基线下降 0.163，MAE 也恶化。更窄的 Innovation–Reaction 方向一致/背离项在 5 日和
20 日分别提高秩相关 0.027 和 0.035，逐一剔除发行人后 20 日 IC 仍为正；但有效开发预测只有
14 个。因此项目只把 Alignment-only 冻结为低置信度外部验证候选，完整 v2 被淘汰。详见
[Conditional Drift Optimization v2](docs/conditional_drift_optimization_v2.md)。

这个候选现在已经完成可执行冻结：特征顺序、标准化参数、Ridge 系数、截距和最后训练标签
截止时间全部写入不可变模型包。63 个训练截止日之后的完整财报事件中，47 个因为本地已经
生成收益而被永久隔离；剩余 16 个事件仅用不含结果列的市场状态与入场前 Reaction 完成了
预先打分，尚未计算任何留出样本指标或盈亏。只有 7/16 完全落在开发样本特征范围内，另有
2 个严重外推，因此协变量重叠、外部验证和实盘闸门仍然关闭。详见
[Conditional Drift Forward Candidate v1](docs/conditional_drift_forward_candidate_v1.md)。

预注册时间留出检验现已完成，候选没有通过。对严格预先打分的 16 个事件，Alignment-only 的
20 日 Spearman IC 为 **-0.106**，基线为 **-0.074**；增量 IC 为 **-0.032**，发行人聚类
95% 区间为 **[-0.367, +0.212]**。候选 Top–Bottom 三分位差为 **-13.50 个百分点**，七项
冻结闸门只通过一项。端点复核确认两个超过 50% 的尾部收益来自连续 20 个交易日、非 OTC、
且没有拆股或分红的价格路径，并非简单端点错配。最终结论为
`candidate_effectiveness_not_established`；这 16 个结果从现在起永久转为已观察证据，不会再
被重复当作验证样本。详见
[Conditional Drift Temporal Holdout Evaluation v1](docs/conditional_drift_holdout_evaluation_v1.md)。

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
sec_data/                  # SEC 抓取、PIT 清洗、验证扩展和行情适配器
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

8. **未观察验证样本扩展。** 2022–2025 年的无收益选择登记表包含 2,503 个新候选事件。
   仅使用 SEC 数据可得到 87 家未观察发行人的 1,074 个完整事件；当前 Basic 行情窗口只
   保留 319 个，因此项目选择扩大冻结样本并寻找更长行情，而不是降低模型门槛。

9. **管理层指引 expectation。** 在 35 个精确同季度收入指引事件上，既往管理层指引把
   MAPE 从点时季节性基准的 13.73% 降至 2.81%。发行人聚类和留一发行人稳健性等预测 Gate
   全部通过；独立收益样本已扩大到 23 个事件 / 8 家发行人。描述性 Spearman 为 -0.0069，
   但负向 Innovation 只有 3 个，正式推断 Gate 仍未通过，因此不宣称 alpha。

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
uv run python -m scripts.build_validation_expansion
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
- 第一轮未观察扩展虽然候选事件足够，但财报特征完整的发行人只有 87 家；当前 Basic 行情
  也只覆盖六个季度。
- 应用使用确定性文字模板，不依赖 OpenAI。

软件采用 [MIT License](LICENSE)。第三方数据仍受其原始条款约束。
