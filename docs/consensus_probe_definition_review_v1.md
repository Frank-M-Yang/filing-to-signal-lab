# First-batch calendar and metric review / 首批财季与收入口径核验

Review date: 2026-09-15 UTC. This is a manual source-definition review for the
fixed PSX, MPC, SNEX, DELL probe slice. It is not a provider definition contract,
an actual-revenue join, or authority to admit events. No target actual or return
was used for this review. Public historical filings were consulted for calendar
policies and statement line definitions only.

Follow-up: the [DELL snapshot-jump case](dell_consensus_jump_review_2026_09_19.md)
adds a 2026-09-19 source review. The original observations below remain unchanged;
the follow-up does not grant event admission or silently correct the old snapshot.

## 结论

首批接口响应能抓取，但“有一个 revenue 数字”还不足以直接计算超预期。
需要先确认预测与实际值指向同一财季、同一单位和同一收入口径。
本轮找到具体的待核验项，全部继续保留为未验证；原 37 个注册事件不变。

| 公司 | 2026-09-14 首次抓取的候选情况 | 口径或日期检查 | 处理 |
| --- | --- | --- | --- |
| PSX | 2026-09-30、2026-12-31 两行通过基础字段检查 | 收入政策涉及买卖合同净额列报、税和运费；供应商未说明自己的对应范围 | 保留候选，等待明确的预测—实际口径映射 |
| MPC | 2026-09-30、2026-12-31 两行通过基础字段检查 | 销售及其他经营收入，不等于收入及其他收益总额 | 不按金额“看起来接近”来挑实际值 |
| SNEX | 两个未来季度的收入与分析师数量均为 0，没有有效季度候选 | 总收入与扣除实物商品销售成本后的经营收入有区别 | 保留拒绝记录；不能拿年度预测补季度 |
| DELL | 2026-10-31 一行通过基础字段检查 | 周制财年与供应商月末日期需要对应；本地 submissions 顶层财年字段也有冲突 | 不自动改日期，不注册新事件 |

这张表描述的是首次快照，不声称已经取得第二次观察。最新次数、真实时间间隔
及逐对字段变化，以本地生成的 coverage probe 审计为准。

## Calendar evidence: DELL

Dell states that its fiscal year lasts 52 or 53 weeks and ends on the Friday
nearest January 31. Its fiscal 2026 ended on 2026-01-30. The provider candidate
uses 2026-10-31, a Saturday. This is a **calendar-mapping question**, not proof of
a wrong estimate. We must not silently round to Friday or invent a replacement
period. [Dell fiscal 2026 Form 10-K, fiscal-year policy](https://www.sec.gov/Archives/edgar/data/1571996/000157199626000008/dell-20260130.htm).

The existing, hash-checked local submissions snapshot has `fiscalYearEnd=1231`.
Its own filing rows nevertheless include these dates:

| Form | Report period | Accession | SEC acceptance time (UTC) |
| --- | --- | --- | --- |
| 10-K | 2026-01-30 | 0001571996-26-000008 | 2026-03-16 20:12:39 |
| 10-Q | 2026-05-01 | 0001571996-26-000030 | 2026-06-09 20:11:41 |
| 10-Q | 2025-10-31 | 0001571996-25-000127 | 2025-12-09 21:07:24 |

The snapshot hash matches its archived filename and appears in the normalized
bundle's source hash list. This confirms local integrity, not universal correctness
of every metadata field. The snapshot was retrieved on 2026-08-13; it cannot prove
the absence of a later filing. Do not use its top-level year-end field alone to
construct fiscal periods. Do not use the acceptance times above as availability
timestamps for the future target quarter.

## Revenue-definition evidence

MPC reports a sales/operating-revenue line separately from a larger total that
also includes equity-method investment income, asset-disposal gains and other
income. A generic provider label does not identify which line was forecast.
[MPC 2025 Form 10-K, consolidated statements of income](https://www.sec.gov/Archives/edgar/data/1510295/000151029526000009/mpc-20251231.htm).

StoneX describes operating revenue as total revenue less physical-commodity
cost of sales. The two quantities are not interchangeable merely because both
may be described as revenue. This historical definition is a review reference,
not confirmation of the forecast provider's current definition.
[StoneX segment disclosure](https://www.sec.gov/Archives/edgar/data/913760/000091376025000147/R26.htm).

Phillips 66 describes net treatment of paired buy/sell inventory contracts in its
purchased-products expense line, and customer freight charges within operating
revenue. Its tax policy also specifies a separate presentation. These details
need a documented provider mapping; the review does not infer a blanket tax
inclusion/exclusion rule or make numerical adjustments.
[Phillips 66 2025 accounting policies](https://www.sec.gov/Archives/edgar/data/1534701/000153470126000006/R9.htm).

Alpha Vantage documents quarterly/annual EPS and revenue estimates, counts and
revision history. The endpoint description reviewed does not explicitly resolve
the currency, scale, accounting scope or analyst identities for our captured
revenue rows. Documentation for a different endpoint cannot fill this gap.
[Alpha Vantage Earnings Estimates documentation](https://www.alphavantage.co/documentation/#earnings-estimates).

## What would resolve the open mappings?

Retain a dated provider definition or response field establishing currency,
scale, accounting basis and fiscal-period convention. Match it to a specific
original filing line, tag/context, unit and period, preserving both sources.
Any manual mapping needs a reason, evidence hash, review timestamp and an explicit
review state. Approximate numerical agreement alone is not a definition test.
If this evidence remains unavailable, keep the rows as isolated coverage evidence
or separately labelled exploratory observations; do not promote them to the
confirmatory sample or silently reinterpret their values.

For the repeat-capture step, record actual elapsed hours and source versions.
Two UTC dates separated by minutes are not two full days of observation. Neither
24 hours of elapsed time nor changed aggregate fields establishes independent
analyst information. A genuine mean change can also reflect panel composition.

## Local source fingerprints

The following are hashes of cached submissions metadata, not companyfacts or
newly fetched filing bodies. All were checked against their existing filenames.
No private data paths or credentials are needed in this review.

| CIK | SHA-256 |
| --- | --- |
| 0001534701 (PSX) | d6a0773a04dbd86028a898e7be6a52a1622af3451b7b4ff183abddd9895d0b15 |
| 0001510295 (MPC) | e2964a7215df5bd7fd8ff0e28424fad273900f1de62448077e54b0e93b27065c |
| 0000913760 (SNEX) | 3942571dea4a59e7e58cccdd92b0e77a3510b91ecc63a70a82324958cd62f964 |
| 0001571996 (DELL) | 1458f9935e361eb36cd204c3e47538c445ae2993e4bcd05638fd2326b01a00bd |

This note is a source-linked manual review, not an immutable automated mapping
dataset. Links were consulted on the review date; target actuals and prices were
not loaded into the coverage or continuity pipeline.
