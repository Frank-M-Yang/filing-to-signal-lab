# Historical revenue consensus sample request — not sent

此文为申请草稿，不代表已联系供应商。发送前由用户决定收件渠道、身份及可接受的
使用条款。没有用户邮箱、账号、API 密钥或订阅信息；请勿附密钥。

候选渠道：

- FactSet：其[官方 PIT 方法页](https://insight.factset.com/resources/at-a-glance-factset-estimates-point-in-time-consensus)
  列有样例演示/销售联系入口，产品名称应明确为 Estimates Point-in-Time Consensus。
- LSEG：使用[官方页面的 Request details 入口](https://www.lseg.com/en/data-analytics/asset-management-solutions/portfolio-management/backtest-your-portfolio-performance)，
  明确询问 I/B/E/S Point-in-Time，而不只询问普通 historical estimates。

Subject: Small evaluation sample — point-in-time quarterly revenue consensus

Hello,

We are evaluating a historical point-in-time revenue-consensus dataset for a
small local research project. We would like to validate definitions and historical
availability before considering any subscription or migration.

Could you advise whether your point-in-time product can provide an evaluation
sample for the following fixed issuer list: AMZN, WMT, UNH, AAPL, MCK, GOOGL, PSX,
MPC, SNEX and DELL?

Our request pack specifies two exact SEC 10-Q report periods per company and
two historical perspective times per period: 30 and 7 calendar days before the
period end, at 23:59:59 UTC. This is 40 requested observations in total. For each,
we request the latest native point-in-time snapshot no later than the requested
perspective time. Please do not substitute a current forecast for a historical
vintage, an annual estimate for a quarterly one, or another company for a missing
observation. Please preserve explicit missing/unsupported results.

Please supply or document:

- Provider issuer/security identifiers, the mapping to CIK and exact fiscal period,
  and treatment of normalized month-end labels and weekly fiscal calendars.
- Standard consensus revenue mean and contributor count, available range statistics,
  currency, scale, accounting definition, averaging/window and stale-estimate rules.
  For this first comparison, we are not requesting a proprietary Sharp/Smart signal.
- Native snapshot time, timezone and granularity; historical activation, availability,
  correction and currency-change semantics. Daily PIT can be suitable: we do not
  require individual analyst timestamps if aggregate PIT availability is documented.
- Original field names, field dictionary, extraction parameters and raw sample files.
- Definitions linking the forecast to an appropriate reported/comparable actual.
  Please do not include realized revenue values or stock returns in this first sample;
  we are validating the source, not selecting it using performance.
- Evaluation/local-research permissions and restrictions on redistribution, derived
  research results, and future access. Please identify the exact product and any
  separate entitlements required.

This is a deliberately small definition/calendar stress test, not a representative
universe or a request for a full database. The request pack contains stable request
IDs and source metadata. Please let us know what sample delivery is available and
any conditions before activating a service or incurring charges.

Thank you.

## 发送时的附件说明

可审阅后附 `consensus-source-trial-v1-cf31c55bd0f0a2d2.json` 请求包，或仅发送其中
`requests` 列表与本草稿。请求包只有身份/日期/公开来源及状态，不包含本地密钥、
用户邮箱、股价、实际收入或策略结果。不要附 SEC/供应商原始缓存整目录。
