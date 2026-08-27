# Unseen Validation Expansion v1

## English

### Purpose

This round expands the return study without recycling outcomes that were already inspected. The
selection unit is frozen before any new return is loaded, collection failures cannot be replaced,
and provider coverage cannot change cohort rank.

The plan uses the existing deterministic, asset-size-stratified SEC probability sample. It then
follows selected issuers longitudinally from 2022 through the sealed-holdout boundary. The
original probability weights do **not** estimate a population event effect after this
issuer-level longitudinal expansion; this is a model-development cohort.

### Frozen evidence roles

The pilot and longitudinal ledgers contain outcomes from 28 issuers. Every new event belonging to
one of those issuers is conservatively labelled `legacy_issuer_training_only`, even when that
exact quarter was not previously scored. Events from all other selected issuers are labelled
`unobserved_validation_candidate`. The expansion builder reads only identifiers and timestamps
from the legacy ledgers, never return values.

### First expansion result

| Stage | Events or units | Issuers |
|---|---:|---:|
| Provisional equity sample | 222 issuer-years | 218 |
| Original 10-K/10-Q events, 2022–2025 | 2,924 | 208 |
| Unobserved validation candidates | 2,503 | 180 |
| Complete unobserved Level/Delta/Innovation | 1,074 | 87 |
| Complete unobserved features inside current Basic price window | 319 | — |

The filing feature build reconstructed 3,246 quarterly point-in-time observations. Among all
2,924 targets, 1,279 have complete standardized Innovation; 1,074 of those belong to unobserved
issuers. No market return was loaded.

The result identifies two independent bottlenecks:

1. **Sample breadth:** 1,074 complete unobserved filing events from 87 issuers do not satisfy the
   frozen 2,000-event/100-issuer training gate.
2. **Price history:** the documented Massive Basic window starts on 2024-08-14. Only 319 complete
   unobserved filing events fall inside it, across six calendar quarters. This cannot satisfy the
   frozen eight-quarter validation gate.

Completing the remaining 188 reference queries is useful for a larger recent-price pilot, but it
cannot by itself authorize the model tournament.

### Stage-two sample freeze

The next SEC-only layer raises the deterministic allocation from 10 to 30 issuer-years per each
of 50 year × point-in-time asset-size strata. It preserves all original 500 units and adds the
next 1,000 pre-ranked units without replacement. The frozen plan contains 1,500 issuer-years and
1,411 unique CIKs. Equity status and feature coverage for the added units remain unknown until
their filing-cover evidence is collected.

Build the plan:

```powershell
uv run python -m scripts.build_sec_probability_sample `
  --universe-root data/processed/sec_universe_design/sec-universe-design-v6-df32ee21a25f59ca `
  --sample-per-stratum 30
```

Collect the 1,000 newly added SEC filing covers in resumable batches. Keep the User-Agent in the
local shell; never write it into repository files:

```powershell
$env:SEC_USER_AGENT = (Get-Clipboard).Trim()
uv run python -m scripts.collect_sec_probability_sample `
  --sample-plan-root data/processed/sec_probability_sample/sec-probability-sample-plan-v1-536f4a7fa7c6657d `
  --limit 100
```

Repeat the collection command until the new sample is complete. The subsequent scope audit will
freeze eligible equity units before any provider or price result is inspected.

### Rebuild the first expansion

For the already completed 222-unit layer:

```powershell
uv run python -m scripts.build_validation_expansion
```

The command writes an immutable cohort registry, candidate-event registry, collection waves,
sample funnel, quarterly filing history, feature audit, and rejection ledger. The first full
feature build is intentionally expensive; rerunning the same immutable source version reuses the
completed output.

When the future expanded return ledger exists, its Tradeable Dataset build must point back to the
pre-outcome registry:

```powershell
uv run python -m scripts.build_tradeable_dataset `
  --ledger-root <expanded-ledger-root> `
  --price-root <expanded-price-root> `
  --benchmark-root <expanded-benchmark-root> `
  --action-root <expanded-action-root> `
  --preregistration-root data/processed/tradeable_dataset_preregistration/tradeable-dataset-preregistration-v1-539ad5f7cba10371 `
  --evidence-registry-root data/processed/validation_expansion/validation-expansion-v1-8014e6a64dfa6a3f
```

The builder rejects any return event whose accession was not frozen in that registry. This keeps
new outcomes eligible for validation instead of accidentally reclassifying them as legacy data
after the returns are calculated.

### Next gate

After the stage-two SEC scope audit, run a small provider-identity wave before committing to the
entire market-data budget. Historical prices must then come either from a plan/source covering at
least 2022–2025 or from two additional future quarters; the model gate must not be lowered to fit
the current Basic entitlement.

---

## 中文

### 目的

这一轮扩大收益研究，但不重复利用我们已经看过的结果。样本在读取任何新收益之前冻结；
抓取失败不能换公司，数据供应商是否容易覆盖也不能改变样本排名。

方案从已经冻结、按点时资产规模分层的 SEC 概率样本出发，再纵向跟踪入选发行人在 2022
年至封存测试边界之前的财报事件。扩展成发行人纵向样本后，原来的抽样权重不再能估计全
市场事件效应；这是一套模型开发样本。

### 证据角色

此前两个收益账本涉及 28 家发行人。为了保守处理，这 28 家公司的所有新增季度都标记为
`legacy_issuer_training_only`，即使某个具体季度此前没有计算过。其他入选公司的事件才
标记为 `unobserved_validation_candidate`。扩展程序只从旧账本读取 CIK、accession 和
时间戳，不读取任何收益数值。

### 第一轮扩展结果

| 阶段 | 事件或单位 | 发行人 |
|---|---:|---:|
| 暂定普通股样本 | 222 个发行人年度 | 218 |
| 2022–2025 年原始 10-K/10-Q | 2,924 | 208 |
| 从未观察的验证候选 | 2,503 | 180 |
| Level/Delta/Innovation 全部完整的新事件 | 1,074 | 87 |
| 同时落入当前 Basic 行情窗口 | 319 | — |

财报层共重建 3,246 个季度点时记录。2,924 个目标中有 1,279 个拥有完整标准化
Innovation，其中 1,074 个来自未观察发行人；整个过程没有加载市场收益。

这次结果找到了两个相互独立的瓶颈：

1. **样本宽度：** 1,074 个完整新事件只来自 87 家发行人，尚未达到冻结的 2,000 事件、
   100 发行人训练门槛。
2. **行情历史：** 当前 Massive Basic 的已记录历史起点为 2024-08-14。只有 319 个完整
   新事件落在窗口内，而且只覆盖六个季度，无法满足八季度验证门槛。

因此，把剩余 188 个证券身份查询全部抓完，能够扩大近期行情试点，但单独做这件事仍不足
以批准模型竞赛。

### 第二阶段样本冻结

下一层免费 SEC 扩展把 50 个“年份 × 点时资产规模”分层的确定性配额从每层 10 个提高到
30 个。原来的 500 个单位全部保留，再按预先冻结的顺序加入 1,000 个单位，不允许替换。
新计划共有 1,500 个发行人年度、1,411 个唯一 CIK。新增单位的普通股资格和财报特征覆盖
必须等 filing cover 抓取后才能知道。

生成计划：

```powershell
uv run python -m scripts.build_sec_probability_sample `
  --universe-root data/processed/sec_universe_design/sec-universe-design-v6-df32ee21a25f59ca `
  --sample-per-stratum 30
```

把新增的 1,000 个 SEC filing cover 分批抓取。User-Agent 只放在本地终端，不写入仓库：

```powershell
$env:SEC_USER_AGENT = (Get-Clipboard).Trim()
uv run python -m scripts.collect_sec_probability_sample `
  --sample-plan-root data/processed/sec_probability_sample/sec-probability-sample-plan-v1-536f4a7fa7c6657d `
  --limit 100
```

重复运行直到新样本完成。随后的普通股范围审计会在查看任何供应商覆盖率或收益之前冻结
合格单位。

### 重建第一轮扩展

对已经完成的 222 单位样本运行：

```powershell
uv run python -m scripts.build_validation_expansion
```

命令会生成不可变的发行人登记表、候选事件表、采集波次、样本漏斗、季度财报历史、特征
审计和拒绝账本。首次全量特征构建计算量较大；相同源版本再次运行会直接复用完成的产物。

未来扩展收益账本生成后，Tradeable Dataset 必须显式指回这份收益计算前的登记表：

```powershell
uv run python -m scripts.build_tradeable_dataset `
  --ledger-root <expanded-ledger-root> `
  --price-root <expanded-price-root> `
  --benchmark-root <expanded-benchmark-root> `
  --action-root <expanded-action-root> `
  --preregistration-root data/processed/tradeable_dataset_preregistration/tradeable-dataset-preregistration-v1-539ad5f7cba10371 `
  --evidence-registry-root data/processed/validation_expansion/validation-expansion-v1-8014e6a64dfa6a3f
```

任何 accession 如果没有提前冻结在登记表中，构建器都会直接拒绝。这样新结果不会在收益
计算完成后又被错误登记成旧数据，从而失去验证资格。

### 下一道门

第二阶段 SEC 范围审计完成后，先做一个小规模证券身份波次，再决定是否投入全部行情预算。
历史行情要么来自能够覆盖至少 2022–2025 年的数据方案，要么等待再积累两个未来季度；
不能为了迁就当前 Basic 权限而降低已经冻结的模型门槛。
