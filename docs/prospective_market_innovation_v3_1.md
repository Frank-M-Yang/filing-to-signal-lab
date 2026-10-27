# Prospective Market-Innovation v3.1 — Earliest-Disclosure Clock Amendment

## English

### What changes and what does not

v3 times every registered company-quarter at the SEC acceptance of its 10-Q/10-K: the
last consensus before acceptance is the expectation, staleness is measured at acceptance,
and entry is the second session after acceptance. The
[first-public clock audit](first_public_clock_v1.md) showed this is structurally wrong for
most issuers: 33.7% of 10-Qs and 62.5% of 10-Ks follow an Item 2.02 earnings 8-K by at
least three days. Under v3, a consensus captured between the release and the periodic
report would be treated as fresh pre-event expectation although it is post-announcement
data; a capture made just before the release could be judged stale at a 10-K accepted
weeks later.

v3.1 changes **one thing**: the event clock.

| | v3 (frozen 2026-08-29) | v3.1 (amendment) |
|---|---|---|
| Event clock | SEC acceptance of the 10-Q/10-K | Earliest SEC-evidenced disclosure: the earlier of the first qualifying Item 2.02 8-K and the periodic report |
| Pre-event consensus | Last capture strictly before acceptance | Last capture strictly before the disclosure **and** no later than the release event date (New York) |
| Staleness | 7 calendar days at acceptance | 7 calendar days at disclosure (unchanged limit) |
| Captures between release and periodic report | Eligible | Counted and excluded |
| Filing monitor scope | 10-Q/10-K | 10-Q/10-K plus original 8-K Item 2.02, read from the same preserved snapshots |
| Entry / exit | 2nd session after acceptance date / 20th close | 2nd session after disclosure date / 20th close |
| Cohort, thresholds, estimand, firewall, retired models | — | **Unchanged**: 37 events, 20 issuers, 30/15/8/8, Spearman IC, no replacement |

Qualifying release: an original 8-K containing Item 2.02, accepted after the issuer's
previous original periodic report, on or after the New York day following the fiscal
period end, within 120 days of the period end, and no later than the period's periodic
report. The rule and its forward implementation mirror the historical audit.

### Declared before any outcome

The amendment record is built by `scripts/freeze_prospective_market_innovation_v3_1`,
which fails closed unless the registry, the readiness ledger, and the event state machine
all show zero actual values, zero innovations, and zero paper positions. It binds the exact
registry manifest, the 37 event identities, the rule text, and the clock and quality bundle
versions at declaration. At declaration (2026-10-27) every event was in
`consensus_revision_ready`; the preserved SEC snapshots end on 2026-09-19 and contain no
target-period disclosure, so the amendment view reports 37/37 `awaiting_disclosure` and
zero disclosed events.

### First live observation (2026-10-27, after declaration)

The first capture after declaration read 39 preserved submissions snapshots with zero
additional requests and found six disclosed events. Clocks and statuses only; no actual
value or return was loaded.

| Ticker | Period | Clock source | Release → periodic lead | Last pre-disclosure capture | v3.1 status | v3 status |
|---|---|---|---:|---|---|---|
| COST | 2026-08-31 | Item 2.02 8-K (Sep 24) | 12.2 days to the 10-K | Sep 19, 5 days | `clean_pre_disclosure` | quarantined (18 days stale at the 10-K) |
| CNC | 2026-09-30 | periodic report (Oct 26) | — | Sep 19, 37 days | `stale_at_disclosure` | quarantined |
| COR | 2026-09-30 | Item 2.02 8-K (Oct 21) | 10-Q not yet filed | Sep 19, 32 days | `stale_at_disclosure` | clean; **would select the Oct 27 post-release capture** |
| ELV | 2026-09-30 | Item 2.02 8-K (Oct 21) | 0.3 days | Sep 19, 32 days | `stale_at_disclosure` | quarantined |
| META | 2026-09-30 | Item 2.02 8-K (Oct 21) | 0.2 days | Sep 19, 32 days | `stale_at_disclosure` | quarantined |
| UNH | 2026-09-30 | Item 2.02 8-K (Oct 13) | 10-Q not yet filed | Sep 19, 24 days | `stale_at_disclosure` | quarantined; **would select the Oct 27 post-release capture** |

Two readings follow directly from the frozen rule. COST is clean under v3.1 because its
September 19 capture was five days before the September 24 release, while v3 quarantines
it as eighteen days stale at the October 7 10-K: the clock change turns one event from
lost to admissible without any reference to its outcome. COR and UNH show the
contamination the amendment exists to prevent: under v3 the latest capture before a
not-yet-filed 10-Q is the October 27 capture, which postdates both releases. The five
September-quarter quarantines are the pre-declared consequence of the capture gap, not a
judgment about those issuers. CNC's clock is its 10-Q because no original 8-K in its
snapshot carries Item 2.02; the rule does not infer a release from other items. The
same run exposed two latent v3 code paths that had never executed on real data (the 10-K
Q4 derivation and the market-evidence audit with queries but no captures); both are fixed
with regression tests.

Later the same day (15:47 UTC, after the 14:25 UTC declaration) the accession-level
companyfacts for COST, CNC, ELV, and META were collected and the v3 ledger computed four
Raw Innovations. For these four events both clocks select the same September 19
capture, because every later capture postdates the periodic report as well as the
release, so the Innovation values are identical under v3 and v3.1; only admission
differs. v3 quarantines all four as stale at acceptance, v3.1 admits COST. No return is
loaded, no standardized Innovation exists yet, and no shadow order was created.

### Pre-declared handling of the capture gap

Daily capture stopped on 2026-09-19. The rule below is fixed now so that no event is
classified by its outcome later:

- An event whose last pre-disclosure capture is more than seven New York calendar days
  before its disclosure is `stale_at_disclosure` and is quarantined **by the rule**.
- An event with no pre-disclosure capture is `no_pre_disclosure_consensus` and quarantined.
- Captures on or after the disclosure are excluded from the expectation and reported as
  `post_disclosure_capture_dates`; they do not disqualify an otherwise clean event.
- No event is replaced, re-registered, or dropped from the registry.

By period end: 1 event (2026-08-31), 15 (2026-09-30), 4 (2026-10-31), 1 (2026-11-30),
15 (2026-12-31), 1 (2027-01-31). Issuers reporting the September quarter announce in
October and early November; the rule above will quarantine those whose last capture is the
September 19 snapshot. Which events that affects is a consequence of calendars and the
rule, and is not reviewed after actuals exist. Even if every later-period event stays
clean, at most 21 events can be admissible this round, below the 30-event minimum. The
round therefore cannot pass the formal gate; its value is an uncontaminated forward ledger
and a method that is correct for the next cohort.

### v3 keeps running

The v3 quality, readiness, state-machine, and shadow bundles are computed and reported
unchanged. The v3.1 view reads them and adds two bundles to the daily run:

- `forward_disclosure_clock`: per-event earliest disclosure from preserved snapshot bytes;
  zero network requests.
- `prospective_disclosure_quality`: pre-disclosure eligibility, staleness, excluded
  post-disclosure captures, `v3_candidate_post_disclosure` (whether v3 would have selected
  a post-announcement capture), and `downstream_admission` requiring a clean v3.1 clock
  **and** a structurally clean v3 status.

### Not yet switched (follow-ups, rule already frozen)

- `forward_innovation_ledger` still joins actuals to the last consensus before acceptance;
  under v3.1 it must use the last pre-disclosure capture. The rule is frozen here; the
  mechanical switch must land before any return evaluation.
- Shadow execution windows still schedule from `sec_accepted_at`; they must schedule from
  the earliest disclosure date under the same date-based contract.
- The v1 state machine is unchanged; a disclosure-aware state (`disclosure_observed`
  before `filing_observed`) is a separate versioned change.

### Boundaries

- Both clocks are SEC proxies. Item 2.02 permits furnishing up to four business days after
  a public announcement; the event-date rule bounds, but does not remove, that residual.
  `first_public_certified` stays false.
- The clock changes nothing about which financial values exist or when: actuals still come
  from the periodic report.
- No actual, price, or return is loaded by either new bundle. Model and trading admission
  stay closed.

### Reproduce

```powershell
uv run python -m scripts.build_forward_disclosure_clock
uv run python -m scripts.build_prospective_disclosure_quality
uv run python -m scripts.freeze_prospective_market_innovation_v3_1
uv run pytest tests/test_forward_disclosure_clock.py tests/test_prospective_disclosure_quality.py tests/test_prospective_market_innovation_amendment.py
```

The daily runner (`scripts.run_forward_research_daily`) now builds both views after the v3
quality audit. The synthetic tests cover release-before-report, same-New-York-date and
late-furnished 8-Ks, amendments, missing clocks, staleness, the outcome firewall, and the
refusal to freeze once any event has an actual.

## 中文

### 改了什么、没改什么

v3 把每个注册事件定在 10-Q/10-K 的 SEC 接受时间：接受前最后一次一致预期是 expectation，时效按
接受时间算，入场是接受后第二个交易日。[首次公开时钟审计](first_public_clock_v1.md)表明这对多数
公司是结构性错误：33.7% 的 10-Q、62.5% 的 10-K 比财报 8-K 晚至少 3 天。按 v3，发布和定期报告之间
采集到的预期会被当成"新鲜的事前预期"，其实是公告之后的数据；而发布前一刻的采集，又可能因为 10-K
几周后才接受而被判为过期。

v3.1 **只改一件事**：事件时钟改为最早的 SEC 披露（财报 8-K 与定期报告中较早者）。事前预期必须在
披露接受之前、且采集的纽约日期不晚于发布的事件日；7 天时效限额不变，但按披露日计算；发布与定期报告
之间的采集计数并排除；申报监控从已保存的同一批快照中读取 8-K 2.02，不新增请求；入场改为披露日之后
第二个交易日。**样本、门槛（30/15/8/8）、主统计量、防火墙、退役模型全部不变，不替换任何事件。**

### 在任何结果出现之前声明

修订记录由 `scripts/freeze_prospective_market_innovation_v3_1` 生成；只要注册表、就绪账本或状态机
里出现任何实际值、Innovation 或模拟仓位，它就拒绝冻结。记录绑定注册表 manifest、37 个事件 ID、规则
原文以及声明时的时钟/质量产物版本。声明时（2026-10-27）37 个事件全部处于 `consensus_revision_ready`；
已保存的 SEC 快照截至 2026-09-19、不含任何目标财季披露，所以修订视图报告 37/37 `awaiting_disclosure`。

### 对采集断档的预先处理

每日采集自 2026-09-19 停止。规则现在定死，以后不按结果分类：最后一次事前采集距披露超过 7 个纽约
日历日的事件按规则隔离；没有事前采集的事件隔离；披露之后的采集排除但不取消事件资格；不替换、不
补注册、不剔除。9 月财季的 15 家公司在 10 月至 11 月初公告，其中最后一次采集停留在 9 月 19 日的
事件会被规则隔离——具体是哪些由日历和规则决定，有了实际值之后不再复核。即使后续财季全部干净，本轮
最多 21 个事件可入样，低于 30 的下限，**本轮不可能通过正式检验**；它的价值是一份未被污染的前向账本
和对下一批样本正确的方法。

### 尚未切换（规则已冻结，机制待落地）

Innovation 账本仍把实际值和"接受前最后一次预期"配对，须改为"披露前最后一次"；影子执行窗口仍按
`sec_accepted_at` 排程，须改按披露日；v1 状态机未改。这些切换必须在任何收益评估之前完成。

### 边界

两种时钟都是 SEC 代理；8-K 可在公告后最多 4 个工作日提交，事件日规则能压缩但不能消除这部分残差，
`first_public_certified` 保持 false。时钟不改变任何财务数值的来源或可用时间。两个新产物都不加载
实际值、价格或收益；模型与交易准入保持关闭。
