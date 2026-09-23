# SEC Clock Provenance and Targeted Acceptance-Clock Overlay

## English

### Scope and result

This round follows the X/OCUL discrepancies from the
[disclosure review](sec_disclosure_followup_v1.md) back to preserved source bytes.
It traces the 32 prior candidate filings and the two affected target 10-Qs: **34
filings in total**. The selection comes from prior clock failures, not returns.
The original 30-event research sample is not reselected.

Five additional public artifacts were collected successfully: two complete 10-Q
submissions and three 8-K index pages. Two target 10-Q index pages were already
preserved locally. Raw bytes and SHA-256 hashes are retained; contact identity is
not written to source code, configuration, or reports.

The result is **five proposed acceptance-clock corrections**, backed by agreement
between SGML headers and filing-index acceptance fields. The other 29 candidate
filings reproduce the labelled UTC source and agree with their SGML header clocks;
they are controls, not proof that all their publication times are valid.

### What caused the discrepancy?

For all five affected records, the preserved submissions JSON already contains an
ISO timestamp ending in `Z`. The catalog reproduces that labelled UTC timestamp
exactly. However, its clock digits match the Eastern wall time displayed in the
SGML header and SEC filing index, not the corresponding UTC time.

Thus, for these exact records, the discrepancy is already present in the upstream
timestamp representation. It is **not an additional four/five-hour subtraction
introduced by our catalog conversion**. This does not certify every parser path:
timezone-less literals remain unresolved, and a catalog/source mismatch would be
classified separately. It also does not imply that all SEC JSON timestamps should
be shifted. That would damage the 29 consistent controls in this audit.

| Issuer | Accession | Frozen catalog UTC | Header/index-reconciled acceptance proxy UTC |
|---|---|---|---|
| X | 0001104659-25-032901 | 2025-04-08 09:05:05 | 2025-04-08 13:05:05 |
| X | 0001163302-25-000028 | 2025-05-01 16:21:59 | 2025-05-01 20:21:59 |
| X, target 10-Q | 0001163302-25-000030 | 2025-05-02 11:56:15 | 2025-05-02 15:56:15 |
| OCUL | 0001104659-24-118284 | 2024-11-14 07:10:12 | 2024-11-14 12:10:12 |
| OCUL, target 10-Q | 0001558370-24-015613 | 2024-11-14 07:15:22 | 2024-11-14 12:15:22 |

For example, X's May 1 earnings 8-K changes from a catalog interpretation of
12:21 New York time to 16:21 New York time. That can change an intraday/after-hours
classification. No price window or realized return was recalculated here; this
does not establish that any specific strategy's execution date or P&L changes.

### Evidence and uncertainty

Each row retains the bulk ZIP hash, member name/hash, original array row index,
exact `acceptanceDateTime` literal, catalog timestamp, SGML header literal/hash,
index literal/hash, and reconciliation status. Issuer, accession, form and fiscal
period identity are checked before comparing clocks. Index identity is checked
against its accession heading, not merely any accession mentioned in the page.

Eastern wall times are interpreted with `America/New_York`, including seasonal
offsets. Missing, ambiguous, nonexistent or contradictory clocks remain unresolved;
they do not become guessed UTC timestamps. A conflict proposal requires both a
header and an index to agree. These are two representations from the same SEC
source, **not independent dissemination evidence**.

Acceptance remains an operational proxy, not certified public availability. The
SEC distinguishes acceptance from publication and does not guarantee dissemination
lag. No fixed delay is invented to bridge this gap. See the
[SEC timestamp explanation](https://www.sec.gov/about/webmaster-frequently-asked-questions).

### Candidate boundaries

For each affected company, the check scans all catalog Item 2.02/7.01 8-K/8-K/A
filings from fiscal-period end through target 10-Q acceptance date, with one
calendar day of padding at each edge. It then evaluates the original and revised
candidate sets using the individually reconciled clocks. A missing candidate or
target clock prevents a completeness claim.

- X retains its two original candidate accessions.
- OCUL retains its one original candidate accession.
- Both target 10-Q timestamps require correction despite unchanged candidate sets.

This is a bounded catalog search, not an exhaustive IR/newswire or preliminary
announcement search. Unchanged membership does not establish valid historical
forecasts or execution windows.

### Outputs and safe use

The overlay is **a separate, unapplied proposal**. No main catalog, feature table,
forecast, price window, forward registry, trading policy or earlier audit bundle
has been overwritten. The overlay cannot serve as a model-ready dataset.

The [configuration](sec_clock_provenance_v1.json) pins the prior plan, follow-up
audit, submissions ZIP and existing index snapshots. Reproduction with preserved
local assets is offline:

```powershell
uv run python -m scripts.audit_sec_clock_provenance
```

Optional collection is limited to missing frozen evidence and requires an
identified process-scoped `SEC_USER_AGENT`, with a shared 0.25-second limiter:

```powershell
uv run python -m scripts.audit_sec_clock_provenance --collect --limit 5
```

No credentials or datasets are needed for the synthetic safety tests:

```powershell
uv run pytest tests/test_clock_provenance.py
```

Immutable bundles under `data/processed/sec_clock_provenance/` contain
`source_clock_provenance.json`, `proposed_acceptance_overlay.json`,
`candidate_boundary_audit.json`, `missing_artifacts.json`, `report.json` and a
Chinese `readout.md`, with manifest hashes. Distinct raw evidence is never replaced.

The public source/docs/tests candidate crossed the old 5 MiB aggregate test budget
by about 28 KiB during this round. The aggregate budget is now explicitly 6 MiB
and enforced in the release-audit command itself. The existing 5 MiB single-file
limit, private-data path exclusions and all secret/contact scans remain unchanged;
separate tests cover both size limits. No research files were deleted to meet a budget.

### Next step

Implement a versioned clock layer that retains original literals, timezone claims,
raw-header/index checks and unresolved states. Test the effect of an explicitly
approved overlay on event identity, as-of feature joins, forecast cutoffs and
execution-window construction together. Do not shift just a target actual's time
and call the old forecasts revalidated. Complete the remaining release and
accounting-scope reviews before admitting any new experiment.

## 中文

这轮把时间差追到了原始数据，结论比“可能有时区错误”更明确：

- 新补充的**5份文件全部下载成功**，完成**34份文件**的溯源核对。
- X、OCUL涉及的**3份8-K和2份10-Q**都存在同类差异。原始SEC JSON带着UTC标记，
  但其中的钟面数字对应原文和索引显示的美东时间。目录程序没有另外减掉4或5小时。
- **另外29份对照文件时间一致**，所以绝不能把整个数据库统一平移。
- 已生成**5条逐文件时间修正提案**，保留旧值、原始证据和新值，尚未写入主库。
- 重查公告候选后，X仍是2份、OCUL仍是1份，没有因修正时钟而换样本。

为什么重要？例如X的一份业绩公告，原目录会把它解释成纽约中午12:21，原文证据却对应
下午16:21：**可能把收盘后的信息当成盘中已知信息**。这会影响如何设计反应窗口，但本轮
没有重算收益，不能说策略收益已经增加或旧结果一定全错。

这些时间仍然只是SEC接收代理，不是市场首次获得消息的精确时点。后续要把时间修正与
特征、训练截止和交易窗口一起版本化重建；只改一个时间字段，不足以证明模型已无偏。

本轮没有训练模型、没有读取收益、没有修改交易规则，没有新增Alpha结论。
