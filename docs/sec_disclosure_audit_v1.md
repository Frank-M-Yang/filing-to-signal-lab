# SEC Disclosure Clock and Financial-Field Audit v1

## English

### Correction and follow-up: 2026-09-20

The [follow-up review](sec_disclosure_followup_v1.md) supersedes this report's
original **available-by upper-bound** interpretation. SEC acceptance is an
operational proxy, not a guaranteed time of public availability. The old immutable
bundles and review specification are retained as audit history; their clock labels
must not be interpreted as certified publication evidence. New runs use proxy
labels. The updated review also checks raw header clocks and quarantines conflicts.

### Purpose and scope

This is a measurement audit, not a new factor tournament or a return experiment.
It asks whether the existing filing event is also the first disclosure of its
financial actuals, and whether original quarterly cash-flow fields can be used
without mixing cumulative periods or accounting scopes.

The configuration pins the existing market-state, filing-feature and SEC catalog
manifests and the preserved companyfacts ZIP by SHA-256. Only identity and clock
columns are projected from the pre-existing research tables. No return, forecast
value or price is loaded to select events.

From the 227 events with historical price evidence, 177 are original 10-Q events.
The first pass takes 30 events in issuer-round/salted-event-hash order, yielding
30 distinct issuers. It does not screen for current listing status or financial
field completeness. Failures remain in place. The plan is persisted **before**
opening financial actuals or release text. This conditional historical sample is
neither a market-representative sample nor an untouched validation set.

### Three different clock claims

1. An Item 2.02/7.01 8-K between financial period end and 10-Q acceptance is a
   **candidate**. The 8-K report date is not assumed to be the earnings period.
2. Matching an exhibit's period, GAAP scope, unit, table column and number to an
   original 10-Q fact establishes matching content in the earlier accepted filing,
   not a proven public-availability upper bound. Assertions are bound to submission and exhibit
   hashes, text anchors and exact scaled values. Keyword matches are not enough.
3. The acceptance proxy is **not** a certified first-public timestamp. A newswire or IR
   release may be earlier, and SEC dissemination may lag. Dates without times must not acquire invented
   times. Missing candidates do not establish that no prior release existed.

The candidate search is deliberately bounded; it does not cover every website,
newswire, preliminary announcement or other SEC form. An exhaustive first-public
review remains unfinished. Each financial metric retains a separate status.

### Initial local results

- 28/30 events have at least one earlier Item 2.02/7.01 candidate.
- 20/32 distinct candidate submissions are already preserved locally; 12 need
  collection. This run made no network requests.
- The first three cached releases in sample order were manually reviewed, not
  selected by their returns. Seven metric values match their later 10-Q originals.

| Issuer | Financial period end | SEC acceptance proxy (UTC) | Acceptance lead over 10-Q | Matched fields |
|---|---|---|---:|---|
| TTC | 2025-01-31 | 2025-03-06 13:38:37 | 3.89 hours | Revenue, gross profit, operating cash flow |
| USNA | 2025-03-29 | 2025-04-22 20:06:21 | 334.60 hours | Revenue, gross profit |
| MBUU | 2024-09-30 | 2024-10-31 11:02:12 | 2.17 hours | Revenue, gross profit |

Sources are the preserved exhibits of accessions
`0000737758-25-000030`, `0000896264-25-000119`, and
`0001590976-24-000099`. The review specifications and source hashes are in
[sec_disclosure_review_v1.json](sec_disclosure_review_v1.json).

This matters beyond the choice of return window: the old forecast training-end
and generated-at metadata for **USNA and MBUU** fall after the earlier acceptance
proxy. This flags timing risk, but the exact publication/forecast order is not
certified by acceptance alone. TTC precedes the proxy; this also does not certify
its unknown first-public timestamp. This finding is
not a new return result, does not prove how much historical IC changed, and does
not silently rewrite any original experiment, forecast or forward policy.

### Accounting-field coverage

| Field | Direct original quarter/instant values | Remaining issues |
|---|---:|---|
| Revenue | 27/30 | 3 ambiguous tag/value sets |
| Gross profit | 19/30 | 11 without a direct standard GrossProfit fact |
| NetIncomeLoss | 30/30 | Parent versus consolidated scope still requires review |
| Operating cash flow | 13/30 | 17 cross-filing YTD differences requiring basis review |
| Total assets | 30/30 | Balance-sheet instant, not a flow |

These counts are structural coverage, not formal model admission. Missing direct
GrossProfit does not establish that gross margin is unavailable: the existing
pipeline also has revenue-minus-cost derivations. This narrow audit does not
silently substitute them. Revenue alternatives are not added together or chosen
by which value improves coverage.

For cumulative operating cash flow, a candidate subtraction requires the same
tag, USD unit and fiscal-year start, a previous-period end immediately before
the current quarter, and a prior original 10-Q accepted strictly before the
target. Both source records are retained. Because cross-filing revisions or
scope changes could break comparability, the difference is **not admitted**.
Later restatements never fill an absent original.

There is another concrete scope warning: the MBUU release reports consolidated
quarterly net loss of USD 5.147 million, but parent-attributable net loss of
USD 5.048 million; the selected NetIncomeLoss fact is the latter. USNA similarly
reports USD 9.290 million consolidated earnings versus USD 9.402 million
attributable to the parent. A naive net-income-minus-consolidated-operating-cash-
flow feature would mix scopes. No such ratio was calculated.

The 13 directly complete net-income/cash-flow/assets cases are therefore not a
training sample. Restricting to them would also favor quarters with direct
cash-flow reporting, including first quarters. Missing cases cannot simply be
discarded and called representative.

### Collection follow-up: 2026-09-20

After explicit authorization to reuse the SEC contact identity, all 12 missing
submissions were downloaded: **12 completed, zero failed, 32/32 candidate documents
now cached**. The first attempt was blocked by the local network sandbox; a
network-authorized retry succeeded. The original failure records remain as
attempt history, not unresolved missing documents. No paid credentials were used.

The rebuilt bundle is `sec-disclosure-audit-v1-65b7f0b7641e158d`. The frozen
30-event plan is unchanged. Raw bytes and hashes were preserved before parsing.
This closes only the candidate-document collection gap: the three manually
reviewed cases, two forecast-clock warnings, 13 directly complete accounting
cases, and closed model-admission gate are unchanged. First-public timestamps
and the remaining event/metric reviews are still unresolved.

### Reproduction and safety

With the existing pinned local research assets:

```powershell
uv run python -m scripts.build_sec_disclosure_audit --plan-only
uv run python -m scripts.build_sec_disclosure_audit
```

Without these assets, the synthetic tests remain independently runnable:

```powershell
uv run pytest tests/test_disclosure_audit.py
```

Optional collection requires an identified process-scoped `SEC_USER_AGENT` and
an explicit request; it is never needed for the offline command:

```powershell
uv run python -m scripts.build_sec_disclosure_audit --collect --limit 12
```

The collector uses the existing byte-preservation mechanism at a 0.25-second
request interval. Contact identity is not saved in code or audit reports. New
responses never overwrite distinct raw bytes. A missing bulk asset or a hash
mismatch fails rather than fetching a replacement or selecting another company.

Outputs are immutable local bundles under `data/processed/sec_disclosure_audit`:
`plan.json`, `field_audit.json`, `document_review.json`, `clock_audit.json`,
`verified_earlier_values.json`, `report.json` and `readout.md`, with output hashes.
The repository does not include raw research datasets or the generated bundles.

### Stop/go decision

**No model run is authorized.** With all 32 candidate documents now cached, complete
the remaining event/metric reviews; establish first-public clocks or explicitly retain a
filing-time-only interpretation. Before any pre-announcement experiment, rebuild
forecasts with all feature, training and generation clocks before the defensible
availability cutoff. Resolve YTD and parent/consolidated cash-flow compatibility.
Only then freeze a separate cash-flow-quality hypothesis and a new holdout.
Do not change the original forward registry or trading policy in this audit.

## 中文

**措辞更正：**SEC接收时间不是已证实的公开可得时间上界。此前把预测晚于接收时间
直接解释为晚于公开披露，证据不够，应改为“时钟风险待核实”。旧产物保留不覆盖；
最新结论见[附件复核报告](sec_disclosure_followup_v1.md)。

这一轮完成的是本地数据审计与可复现工具，不是新模型训练。固定抽取30家公司各一个原始
10-Q事件，未读收益挑样本，未因缺失换公司，也未改任何原模型或交易规则。

主要发现：

- **28/30**个事件存在更早的相关8-K候选。初轮缓存为**20/32**份；2026-09-20补抓成功12份，
  现已达到**32/32**，没有下载失败项。初次网络权限失败记录仍保留，不代表文件目前缺失。
- 按固定顺序核对了最先有缓存的三家公司：TTC、USNA、MBUU的收入和毛利润出现在较早接收的8-K中；
  TTC还匹配到经营现金流。其中USNA两份文件的接收时点相差约14天，不等于已证实的公开时间差。
- USNA和MBUU原预测的训练结束及生成时点晚于相关8-K接收时间，应标记时钟风险；
  尚不能单凭该代理时钟证明公开先后关系。旧收益结果的影响大小仍待重建时钟后另行检验。
- 30家公司都有原始NetIncomeLoss和资产字段，但只有13个事件直接提供单季度经营现金流；
  另17个需要累计值相减及口径审查。不能把半年或九个月的现金流直接配上一个季度的利润。
- **有数字不等于口径匹配。** MBUU的合并净亏损为514.7万美元，归母净亏损为504.8万美元。
  直接把归母利润与合并经营现金流相减，可能制造虚假的“利润质量差异”。目前未计算此类信号。

因此先保持模型闸门关闭。候选公告原文已经补齐，下一步逐项确认信息时钟和利润/现金流口径，再决定
是否有条件进入新假设实验。旧样本不能包装成新的独立验证，当前没有新增收益或盈利结论。
