# Publisher-time replay and market-feature readiness

## English

### Question and boundary

Can the existing gross-margin expectation and Innovation be reproduced using only
information preceding the publisher's release label, instead of the later 10-Q
acceptance proxy? This is a clock-sensitivity diagnostic, **not an alpha test**.

The earlier source audit fixed this candidate before replay: UFPT, quarter
2025-07-01 to 2025-09-30; revenue USD 154,558,000 and gross profit USD 42,747,000.
The issuer-linked [Nasdaq release](https://www.nasdaq.com/press-release/ufp-technologies-announces-strong-q3-2025-results-2025-11-03)
labels publication **2025-11-03 16:08 EST / 21:08 UTC**. Its exact saved bytes have
SHA-256 `d4b08ed97f6f0b217ebe88f795d3f9affb8d7e5557fa4cea6510c4f040984aa9`.
The 10-Q accession is `0001628280-25-050425`, accepted at
2025-11-07 12:55:24 UTC. Source identity, quarter and both financial cells are
rechecked against the pinned source-review artifacts before any replay.

These pages were retrieved retrospectively in September 2026. A page's publisher
label is **not proof of earliest public dissemination**, a historical capture, or
the absence of an earlier release channel. All first-public, model and return
admission gates remain closed.

### Method

1. Reproduce the existing isolated SEC-clock baseline over all 3,254 accounting
   events, without changing the pooled company-history OLS specification, its
   six lagged predictors, minimum 100 training observations or 50 prior OOS errors.
2. Change only the verified UFPT quarter's `actual_available_at` in a new copy.
   Keep original values, event selection, SEC acceptance, and all YTD/Q4 source
   clocks intact. Quarter-component evidence is not YTD-component evidence.
3. Rebuild all as-of lags, forecast cutoffs, expanding training samples and strictly
   prior out-of-sample errors/sigma across the entire history, not just UFPT.
   Enumerate the target's training and error memberships and reproduce sigma.
4. Replay both 21:08:00 and 21:08:59.999999 UTC to respect minute precision.
   Stop if another event falls inside the minute, if the target forecast uses
   information from that minute, or if endpoint signals/memberships differ.
   These endpoints are a sensitivity range, not certified first-public bounds.
5. Verify that every event strictly before the revised label is unchanged.
   Join market-readiness metadata to **all 316 frozen execution events**; do not
   silently restrict selection to the 227 rows in the existing market store.
   No post-event prices or return values are loaded.

### Results

|UFPT measure|Original filing proxy|Publisher-label sensitivity|
|---|---:|---:|
|Training observations|2,519|2,498|
|Prior OOS errors for sigma|2,419|2,398|
|Reconstructed forecast generated at, UTC|2025-11-06 23:29:08|2025-10-31 20:09:22|
|Actual gross margin|27.657578%|27.657578%|
|Predicted gross margin|28.445781%|28.452973%|
|Raw Innovation, percentage points|−0.788203|−0.795395|
|Standardized Innovation|−0.096857|−0.097391|

Twenty-one observations disappear from the target's training and error sets.
The signal stays negative; its change is small. Neither outcome was imposed or
tuned. All 3,196 earlier events remain unchanged. Across the full history, 22
forecasts and 35 standardized innovations change beyond absolute tolerance
`1e-10`; effects propagate through pooled training and the prior-error history.
The two minute endpoints yield identical numerical outputs within that tolerance,
although dependent timestamp fields retain their scenario-specific precision.

This does **not** show that the timing problem was harmless for returns. Under the
same second-session-after-announcement-date, 20-session-inclusive convention,
the schedule shifts from **November 11–December 9** to **November 5–December 3**.
No old price/action/delisting evidence is automatically valid for the new window.
We did not recompute returns.

This is one deliberately selected timing-risk case, not a random sample of earnings
events. Small changes here cannot establish general robustness, and the remaining
history still uses a mixture of reviewed and unreviewed acceptance proxies.

### Three-arm readiness: coverage is not permission to fit

|Check over the same 316 candidate events / 64 issuers|Count|
|---|---:|
|Financial Level/Delta/standardized-Innovation representation present|316|
|Market metadata present|227|
|Market metadata absent|89|
|Present row lacks sufficient market history/cutoff|52|
|Existing market cutoff is at/after revised event time|1 (UFPT)|
|Joint financial + market metadata passes current proxy-clock checks|174|
|Admitted to the proposed new return experiment|0|

UFPT's old market features use prices through **2025-11-06 21:00 UTC**, after
the release label. They are excluded from joint coverage and placed in a rebuild
queue. The other 174 are **provisional metadata coverage**, not 174 validated
first-public events. Their remaining acceptance proxies still require review;
the market store also has pre-existing availability/selection restrictions.

### Next experiment, only after the timing and window gates

First rebuild UFPT's pre-release market state from timestamped cached prices where
available, without reusing November 6 features. Continue resolving the already
identified USNA/MBUU publication-clock uncertainties; preserve missing cases.
Then verify price, security identity, corporate actions and delisting coverage for
the revised holding windows. Do not make retrieval success a sample-replacement rule.

Before opening new outcomes, freeze the exact common sample, feature columns,
target, cost assumptions, folds, purging policy and primary statistic for:

- **Financial only:** separately evaluate Level, Delta, raw Innovation and
  standardized Innovation; distinguish pre-event State from the newly released value.
- **Market only:** pre-event market/stock momentum, volatility, liquidity and relative
  performance; all cutoffs strictly precede the event.
- **Financial + market:** the same financial representation plus the same market
  block, evaluated on exactly the same events, splits and target as its controls.

The proposed target is the timestamp-explicit 20-session market-excess return, not
gross-margin forecast MAE. Use a simple regularized linear baseline initially;
fit scaling, imputation and hyperparameters on training data only. Calendar-time
walk-forward splits must exclude training labels not fully realized before the
test decision and purge overlapping holding windows; use issuer/time-dependence
aware uncertainty and report missingness, costs and all predeclared comparisons.
This is a design checklist, not a finished preregistration or authorization to train.
Do not add political news, politician transactions or deep learning to this small
timing-repair experiment.

### Reproduction and artifacts

```powershell
uv run python -m scripts.build_publisher_replay
```

No network request or credential is needed **when the pinned local upstream
artifacts already exist**. A fresh clone alone does not contain the private raw
and processed bundles. This command is therefore not a fresh-clone public demo;
see the README public-reproduction path and the synthetic offline tests instead:

```powershell
uv run pytest tests/test_publisher_replay.py
```

`docs/publisher_replay_v1.json` pins manifests, not mutable “latest” paths.
The new immutable bundle under `data/processed/publisher_replay/` contains the
plan-bound report, bilingual readout, both disarmed full-history Parquet panels,
field-level differences, exact target dependency ledgers, the full 316-event
readiness ledger, and a market-rebuild queue. Existing outputs remain untouched.
The forecasts are historical reconstructions, **not forecasts logged live in 2025**.

Validated local result: `publisher-replay-v1-43a52c085513ed1b`. The full suite passes
552 tests; formatting, lint, type checks and the public-release privacy scan pass.

## 中文

### 这一轮做了什么

我们没有添加新因子，而是检验一个基础问题：**在信息刚发布时，模型是否
只使用了当时能知道的数据？** UFPT 的季度营收和毛利润可在标注11月3日的
完整发布页核对，但旧预测训练到了11月6日，原来采用的是11月7日10-Q接收时点。

这次在独立副本中，以发布标签做两端分钟精度敏感性重放，重新计算3,254个
历史事件的滞后特征、预测、训练样本和历史样本外误差。原始财报值、SEC接收
时间、YTD依赖和原研究文件均未改动，也没有调参数来追求某个结果。

### 得到了什么

- UFPT 训练样本从2,519条降到2,498条，去掉了21条晚于发布标签的信息；
  历史预测误差也同步剔除21条，不能只修改表面日期。
- 毛利率预测从28.445781%变成28.452973%，实际值仍是27.657578%。
  原始Innovation从−0.788203个百分点变成−0.795395个百分点：**仍然是略低于
  模型预期，没有方向翻转，数值变化很小**。这不是分析师一致预期惊喜。
- 全历史有22个预测、35个标准化Innovation超过`1e-10`容差发生变化；
  更早的3,196个事件完全不变。两个分钟端点的数值一致。
- 执行窗口却从11月11日—12月9日，移到11月5日—12月3日。
  **信号变化小不代表收益变化小**；这一轮没有读取或重算收益。
- 316个候选事件全部保留，174个通过现有财报/市场元数据的共同覆盖检查。
  89个没有市场行，52个历史不足，UFPT的旧市场特征用了发布后的11月6日行情，
  必须重建。174只是覆盖数，不是已认证可用于新收益研究的样本数；正式准入仍为0。

### 价值与下一步

正面成果是我们已经能把“看似当时可用、实际混入后来信息”的问题定位到具体
记录，并重建整条预测链。**这不是发现了Alpha，更不是证明能赚钱。**
发布标签来自事后保存网页，还不能当作首次公开时间的完整证明。

下一步先重建UFPT公告前的市场状态，继续处理USNA/MBUU时点证据，核对新窗口
的价格、身份、公司行动和退市覆盖。之后再预先锁定同样本、同时间切分、同收益
窗口的“仅财报／仅市场／财报＋市场”对照。暂不加入政客交易披露或深度学习。
