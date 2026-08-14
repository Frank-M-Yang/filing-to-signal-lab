# 28 事件 PIT 财务特征 v1.1

## 目的

本资产专门为真实收益账本中的28个 SEC filing 构建同事件、同信息时钟的 Level、Delta
和 Innovation。原子单位是 `Company × Filing accession × accepted_at`，不是 ticker 加
相近日期。资产只使用冻结的 SEC companyfacts 与 filing catalog，不把最新或修订后的值
回填到历史决策时点。

## 口径

- 10-Q：只选择持续70–110天、与原始 filing accession 绑定的季度 flow facts；
- 10-K Q4：优先使用直接披露的季度 flow；否则使用全年原始值减去更早已披露的原始九个月
  10-Q，且 taxonomy、tag、unit 和期初必须一致；
- Level：该季度的原始毛利率；
- Delta：与上一财年同季度原始毛利率之差；
- Innovation：只用事件前公司历史特征训练 expanding OLS，预测误差为 raw Innovation；
- 标准化 Innovation 的 sigma 只使用严格更早的样本外预测误差，至少50个；
- 第一版不使用行业特征，避免把当前 SIC 误当作历史点时行业分类。

## 冻结结果

- 数据版本：`pilot-filing-features-v1.1-b8c407284c843d77`；
- 历史季度事件：503；
- 历史 Delta：431；
- 历史样本外预测：303；
- 历史标准化 Innovation：253；
- 28个目标 accession 可追溯：28；
- 目标 Level：15；
- 目标 Delta：13；
- 目标 raw / standardized Innovation：11；
- 完整共同样本：11 / 28（39.3%）。

缺失漏斗为：13个 `no_quarterly_level_event`、2个 `delta_unavailable`、2个
`forecast_unavailable`。金融公司、REIT或研发阶段公司可能没有经济上可比的收入和毛利
概念；这些缺失不会被填零。规范化层保留了101,018条拒绝记录及来源哈希，便于逐条复核。

ASNS 的标准化 Innovation 约为 -4.88，是当前11事件共同样本中的明显尾部值。它被原样
保留，但在扩大样本、完成会计口径复核和尾部敏感性分析前，不应据此解释信号—收益关系。

## 研究闸门

收益诊断预先要求至少20个完整事件且覆盖率不低于80%。对28事件而言，覆盖率条件需要
至少23个完整事件；当前11个，因此禁止计算 IC、信号散点或有效性结论。这个结果衡量的是
数据准备度，不是信号失败。

## 复现

```powershell
uv run python -m scripts.build_pilot_filing_features
uv run python -m scripts.build_event_return_diagnostics
uv run streamlit run app.py
```

输出目录包含季度历史、目标特征、目标失败审计、季度选择审计、事实拒绝记录、覆盖表、
合同、报告和 manifest。每个不可变版本都绑定上游 readiness、filing catalog 与
companyfacts 快照 SHA-256。
