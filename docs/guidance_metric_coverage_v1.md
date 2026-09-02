# Multi-Metric Company-Guidance Coverage v1

## English

### Question

If explicit GAAP gross-margin guidance is too sparse, do prior SEC 8-K releases provide enough
point-in-time revenue, diluted-EPS, or operating-margin guidance to support the next Innovation
experiment?

This is a source-coverage and semantic-alignment audit. It does not fit a model and does not load
post-filing returns.

### Design

The audit reuses 169 hash-preserved complete submissions selected for 171 filing events. Every
source 8-K was accepted 30–240 days before the target filing. The parser reads EX-99 attachments,
identifies explicit guidance or outlook sections, and extracts numeric revenue, diluted EPS,
operating margin, and gross margin.

Three distinctions are mandatory:

1. A document may contain a number without providing comparable guidance.
2. GAAP-compatible guidance is separate from adjusted or non-GAAP guidance.
3. Quarterly guidance must name the same fiscal year and quarter as the target filing. Full-year,
   ambiguous, and unstated horizons do not qualify.

Revenue units bind to the statement that contains the value. A different metric expressed in
billions elsewhere in the section cannot rescale revenue stated in millions. Mixed quarterly and
full-year headings remain ambiguous unless the local statement resolves the horizon.

The preregistered pilot gate requires at least 20% event coverage and 10 issuers. Actual target
values and post-selection sanity checks do not feed this gate.

### Coverage results

| Metric | Any numeric events | GAAP-compatible events | Exact-quarter GAAP events | Issuers | Coverage | Gate |
|---|---:|---:|---:|---:|---:|---|
| Revenue | 93 | 82 | 16 | 6 | 9.36% | Fail |
| Diluted EPS | 67 | 39 | 11 | 3 | 6.43% | Fail |
| Operating margin | 21 | 16 | 0 | 0 | 0.00% | Fail |
| Gross margin | 6 | 5 | 4 | 1 | 2.34% | Fail |

The parser found 509 deduplicated numeric observations across 139 documents with identifiable
guidance sections. No metric passed either the evidence-coverage gate or the model-feature gate.
Revenue is the best-supported family, but 16 events from six issuers are not a defensible
cross-sectional training sample.

### Post-selection semantic check

The 16 exact-quarter revenue observations were inspected only after the outcome-blind coverage
selection. Their midpoint mean absolute percentage error is 2.31%, the median is 1.76%, and actual
revenue is on average 2.16% above the guidance midpoint. The reported range contains the actual in
18.75% of cases.

These figures support the semantic validity of the extracted revenue records; they do not
establish a revenue signal. The sample is small, issuer-concentrated, and selected on companies'
voluntary disclosure policy. The sanity statistics are explicitly excluded from metric selection.

### Decision

No forecast model or return diagnostic is authorized in this round. The next evidence investment
should target exact-quarter revenue guidance across a broader issuer set and a longer filing
history. Missing guidance must remain an observed disclosure state, never an imputed number. A
pilot becomes eligible only after reaching at least 20% coverage and 10 issuers on the frozen
cohort. Point-in-time analyst consensus remains the more scalable expectation source if that gate
cannot be reached without a selected issuer universe.

## 中文

### 研究问题

明确的GAAP毛利率指引过于稀疏，那么历史SEC 8-K中的收入、稀释EPS或营业利润率指引，是否
足以支持下一轮Innovation实验？

这一轮是数据源覆盖率和语义对齐审计，不训练模型，也不读取公告后收益。

### 设计

审计复用此前为171个财报事件筛选并保存哈希的169份完整提交文件。每份源8-K的SEC接受时间
都严格早于目标财报30至240天。解析器读取EX-99附件，识别明确的Guidance或Outlook区段，
提取收入、稀释EPS、营业利润率和毛利率数值。

必须区分三个层次：

1. 文件中出现数字，不等于存在可比较的指引。
2. 与GAAP兼容的指引必须和调整后或non-GAAP指引分开。
3. 季度指引必须明确对应目标财报的同一财年和财季。全年、含混或未注明期间的指引均不合格。

收入单位绑定到数值所在的具体表述。同一段落中其他指标使用“十亿美元”，不能把以“百万美元”
披露的收入放大一千倍。若标题同时包含季度和全年，只有局部表述能明确期间时才允许匹配。

预先设定的试验门槛是事件覆盖率至少20%，且至少覆盖10家公司。目标实际值和事后语义校验
都不参与这一闸门。

### 覆盖结果

| 指标 | 含任意数值的事件 | GAAP兼容事件 | 同财季GAAP事件 | 公司数 | 覆盖率 | 闸门 |
|---|---:|---:|---:|---:|---:|---|
| 收入 | 93 | 82 | 16 | 6 | 9.36% | 失败 |
| 稀释EPS | 67 | 39 | 11 | 3 | 6.43% | 失败 |
| 营业利润率 | 21 | 16 | 0 | 0 | 0.00% | 失败 |
| 毛利率 | 6 | 5 | 4 | 1 | 2.34% | 失败 |

139份文件中识别到了指引区段，共得到509条去重后的数值观察。没有任何指标通过证据覆盖闸门
或模型特征闸门。收入是目前最有希望的指标，但16个事件只来自6家公司，不能作为可靠的横截面
训练样本。

### 事后语义校验

只有在不看结果完成覆盖筛选后，才对16个同财季收入记录进行校验。指引中点相对实际收入的
平均绝对百分比误差为2.31%，中位数为1.76%；实际收入平均高于指引中点2.16%；实际收入落在
管理层原始区间内的比例为18.75%。

这些数字说明收入记录的语义和单位基本正确，但不能证明收入信号有效。样本很小、集中在少数
公司，而且由企业自愿披露政策筛选形成。上述校验指标明确不参与指标选择。

### 决策

本轮不授权预测模型和收益诊断。下一项数据投入应扩展同财季收入指引的公司范围和历史长度。
没有指引必须保留为一种真实的披露状态，不能填补成虚构数值。只有在冻结样本上达到至少20%
覆盖率和10家公司后，才允许进入预测试验。如果不构造选择性公司池就无法达到门槛，那么严格
点时的分析师一致预期仍是更可规模化的预期来源。
