# Prior Company-Guidance Evidence v1

## English

### Research question

Can management guidance that was public before a filing provide a scalable point-in-time
expectation input for the gross-margin Innovation model?

This stage is a coverage and measurement audit, not a return backtest. It does not read the
current filing text or any post-filing outcome when selecting prior guidance.

### Point-in-time design

The source cohort contains 175 market-state-ready filing events. For each event, the pipeline
searches the same issuer's SEC filing history from 30 through 240 calendar days before the target
filing acceptance timestamp. It selects the latest Item 2.02 8-K when available, otherwise the
latest Item 7.01 8-K. This deliberately excludes the current earnings release and creates a strict
prior information set.

The design found prior candidates for 171 of 175 events (97.7%), covering 46 issuers and 169 unique
complete-submission files. All 169 files were downloaded with an identified, sub-five-request-per-
second SEC client. Exact response bytes and SHA-256 hashes were preserved before parsing. EX-99
attachments are preferred; the primary 8-K is used only as a fallback.

### Cleaning and measurement

The parser distinguishes forward-looking language from a usable accounting expectation. Safe-
harbor language alone is not called guidance. Numeric gross-margin evidence must occur in a
forward-looking gross-margin statement. GAAP and non-GAAP values are not averaged: when both are
reported, the GAAP value is selected; adjusted-only guidance is labelled non-GAAP and excluded
from the model-ready count. Duplicate attachment text is removed.

### Results

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| Complete-submission collection | 171 / 171 events | 95% | Pass (100%) |
| EX-99 or fallback text extraction | 171 / 171 events | 90% | Pass (100%) |
| GAAP-compatible numeric gross-margin guidance | 5 / 171 events | 50% | Fail (2.92%) |

Seven events contain some numeric forward-looking gross-margin language, but two are adjusted or
otherwise non-GAAP-only. The five GAAP-compatible observations come from only two issuers. This is
far below the preregistered coverage threshold and cannot support a cross-sectional forecast
tournament.

### Interpretation and next step

The ingestion result is strong; the proposed feature is not. SEC 8-K history is available and can
be audited at high coverage, but explicit GAAP gross-margin guidance is a selective disclosure
practice rather than a broadly comparable expectation variable. Training on five observations
would mostly learn issuer identity and disclosure policy.

The pipeline therefore blocks model fitting and return inspection. The next useful test is a
separate coverage audit for more commonly guided targets—revenue, EPS, and operating margin—while
keeping each accounting basis and forecast horizon explicit. A different route is licensed,
point-in-time analyst consensus. Neither source should be merged into Innovation until it can be
matched to the target fiscal period on a complete-case sample.

## 中文

### 研究问题

在当前财报公布之前已经公开的管理层指引，能否成为毛利率 Innovation 模型中可规模化、
严格点时的预期输入？

这一阶段是覆盖率与口径审计，不是收益回测。筛选历史指引时，系统不读取当前公告文本，
也不读取任何公告后结果。

### 点时设计

源样本包含175个市场状态就绪的财报事件。对每个事件，管线只搜索同一公司在目标财报被SEC
接受前30至240个日历日内的历史公告：优先选择最近的Item 2.02 8-K；没有时才退回最近的
Item 7.01 8-K。这样会明确排除当前业绩公告，形成严格的历史信息集。

171/175个事件（97.7%）找到了历史候选，覆盖46家公司、169份唯一完整提交文件。169份文件
全部抓取成功。SEC客户端包含合规身份，并严格低于每秒5次请求。解析前先保存原始响应字节和
SHA-256；优先读取EX-99附件，只有缺失时才回退到8-K正文。

### 清洗与计量

解析器把一般的前瞻性措辞和可用于模型的会计预期分开。只有安全港声明，不能被称为有效
指引。数值毛利率证据必须出现在明确的前瞻性毛利率表述中。GAAP与non-GAAP数值不会被简单
平均：同时披露时选GAAP值；仅有调整后口径的指引被标为non-GAAP，并从模型可用样本中排除。
重复附件文本也会去重。

### 结果

| 闸门 | 实际覆盖 | 要求 | 结果 |
|---|---:|---:|---|
| 完整提交文件抓取 | 171 / 171个事件 | 95% | 通过（100%） |
| EX-99或回退正文提取 | 171 / 171个事件 | 90% | 通过（100%） |
| 与GAAP兼容的数值毛利率指引 | 5 / 171个事件 | 50% | 失败（2.92%） |

7个事件包含某种前瞻性毛利率数值，其中2个仅有调整后或其他non-GAAP口径。最终5个可比观察
只来自2家公司，远低于预先设定的覆盖门槛，不能支撑横截面预测竞赛。

### 含义与下一步

数据管线是成功的，但这个具体特征不成功。SEC 8-K历史可以高覆盖、可追溯地获得；然而，
明确的GAAP毛利率指引是一种选择性披露，而不是普遍可比的预期变量。用5个样本训练，模型主要
会学到公司身份和披露习惯，而不是可推广的经济规律。

因此管线自动阻止模型训练和收益检验。下一步应独立审计更常见的指引目标，例如收入、EPS和
营业利润率，同时明确区分会计口径和预测期间。另一条路线是有许可证、严格点时的分析师一致
预期。任何来源在与目标财季完成时间匹配、形成完整可比样本之前，都不应直接并入Innovation。
