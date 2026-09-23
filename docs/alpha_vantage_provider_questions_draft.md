# Alpha Vantage clarification request — draft, not sent

收件地址：请使用 [官方支持页](https://www.alphavantage.co/support/) 当前列出的支持邮箱（2026-09-19 UTC 已核对）。公开仓库仅保留联系页链接。
以下内容可自行审阅后发送；没有包含用户邮箱、API 密钥、账号或原始响应附件。
不要把密钥附到邮件或聊天。供应商若要求账号核验，请走其官方安全流程。

Subject: EARNINGS_ESTIMATES — revenue definitions, fiscal dates and update timestamps

Hello Alpha Vantage Support,

We are conducting a small, non-commercial, point-in-time financial-data study.
Before interpreting changes in EARNINGS_ESTIMATES as changes in market expectations,
could you clarify the following for the revenue fields specifically?

1. What currency and scale are used for revenue_estimate_average/low/high?
   Are they raw currency units, thousands, or millions? Can currency differ by
   company, listing, or estimate period, and where is that currency documented?
2. What revenue definition is forecast: consolidated reported revenue, net
   revenue, operating revenue, or a provider-adjusted measure? Is there an
   issuer-specific mapping to reported actuals? This matters particularly for
   PSX, MPC and SNEX, where similarly named financial lines differ.
3. For a fiscal-quarter row, is date an actual fiscal period-end or a normalized
   calendar month-end? For DELL, how should 2026-10-31 and 2027-01-31 map to its
   weekly fiscal calendar? Please provide the mapping convention, rather than
   relying on approximate numerical agreement.
4. Can you provide per-period revenue-consensus update/as-of timestamps, including
   timezone and their exact meaning? Does a timestamp refer to analyst publication,
   your ingestion, an aggregate recalculation, or API cache refresh? What refresh
   schedule, cache delay and latency guarantees apply to this endpoint and access tier?
5. Are average/high/low/count calculated over exactly the same analyst set? Is the
   average unweighted? How are stale forecasts, additions and removals handled?
   Are contributor identities or stable anonymous IDs available?
6. We see historical revision fields for EPS in our captures. Is corresponding
   revenue revision history available? Can an earlier consensus vintage be queried,
   or do rows for historical fiscal periods reflect subsequently revised data?
7. Could you help explain this concrete DELL example? Both responses have
   horizon = fiscal quarter and date = 2026-10-31:

   - Retrieved 2026-09-14 23:24:49 UTC: revenue average 40,994,028,220;
     low 34,325,000,000; high 46,001,000,000; analyst count 21.
   - Retrieved 2026-09-19 22:09:23 UTC: revenue average 49,184,510,840;
     low 48,883,706,760; high 49,723,650,680; analyst count 20.

   The later response also adds a fiscal-quarter row dated 2027-01-31. Dell had
   already published Q3 FY27 revenue guidance on September 1. We are not assuming
   that the later capture represents delayed data or identifying a provider error;
   we would like to distinguish analyst revisions, membership changes, fiscal
   remapping, and any provider/cache refresh. Can you establish when this specific
   revenue aggregate was updated and whether its definition stayed unchanged?

Official company source:
https://www.sec.gov/Archives/edgar/data/1571996/000157199626000039/exhibit991earnings8kq2fy27.htm

A field dictionary or written documentation is welcome. If some information is
unavailable, that is also useful: we will keep those interpretations unverified.
We are not requesting a subscription change or purchase.

Thank you.

## 收到回复后的处理

保留原邮件供本地核验，公开报告只放去除个人信息后的摘录。记录收到时间、适用
endpoint、数据层级/套餐及回复适用期间；不要用今天获得的解释伪造过去可得的
逐行时间戳。说明可以解决定义问题，但不必然重建历史版本。
任何入模规则修改都需要另行审查，不会因为一封回复自动放行。
