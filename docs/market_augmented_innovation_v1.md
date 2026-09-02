# Market-Augmented Innovation v1

## English

### Research question

Can market information that was observable before a filing improve the point-in-time forecast of
gross margin, and therefore produce a cleaner accounting Innovation than the existing financial
forecast alone?

This is a financial-forecast experiment, not a trading backtest. Model selection never reads
post-filing returns.

### Point-in-time market state

The input universe is the frozen 227-event authorized return cohort. Price features use only
sessions strictly before the filing calendar date. The feature clock is the official close of the
last included exchange session, which must precede SEC acceptance. Unadjusted prices are converted
to a consistent share basis with corporate actions effective before the cutoff.

The frozen market family contains 5-, 20-, and 60-session price momentum; 20- and 60-session
volatility; short-to-medium volume; dollar volume; SPY momentum; relative momentum; 60-session
beta; and 60-session drawdown. Existing preserved prices provide a complete 60-session state for
175 events from 47 issuers. The missing 52 events are retained as unavailable rather than imputed.

### Forecast tournament

The existing financial forecast is the baseline. Two event-by-event expanding residual-correction
models are compared on the same 134-event, 47-issuer common sample:

| Model | Information | Gross-margin MAE | Change from baseline |
|---|---|---:|---:|
| E0 | Existing financial forecast | 0.01834 | — |
| E1 | E0 plus pre-filing market correction | 0.02057 | +12.16% worse |
| E2 | E0 plus market and prior-accounting correction | 0.02065 | +12.60% worse |

Every correction forecast is trained only on events whose actual gross margin was available before
the current feature cutoff. Standardization uses only strictly earlier out-of-sample errors.

### Result and research gate

Neither market-augmented candidate clears the preregistered 1% relative-MAE improvement gate. E0
remains selected. Because the financial forecast gate failed, the pipeline does not open the return
diagnostic for the rejected candidates.

The result says that simple pre-filing price, volatility, volume, beta, and drawdown features do not
improve gross-margin expectations in this sample. It does not say that market state is useless for
predicting returns; it says that this feature family should not be relabelled as a better accounting
Innovation.

The next expectation inputs should come from information closer to the accounting expectation
itself: prior company guidance, point-in-time analyst consensus, and structured lagged operating
drivers. Each family should be added separately and must pass the financial-forecast gate before
its return relationship is inspected.

## 中文

### 研究问题

公告前已经存在于市场中的信息，能否提高对本期毛利率的点时预测，从而构造出比现有财务
预测更干净的 Innovation？

这是财务预测实验，不是交易回测。模型选择过程完全不读取公告后收益。

### 点时市场状态

输入范围固定为227个已授权收益事件。价格特征只使用严格早于公告日期的交易日，特征时钟
是最后一个纳入交易日的官方收盘时间，并且必须早于SEC接受时间。未复权价格根据特征截止
前已经生效的拆股记录转换到一致股份口径。

冻结的特征包括5、20、60日价格动量，20、60日波动率，短中期成交量比、成交金额、SPY
动量、相对动量、60日beta以及60日回撤。现有保存行情可以为175个事件、47家公司提供完整
的60日状态；缺失的52个事件被保留为不可用，没有填补。

### 预测竞赛

现有财务预测作为基线，在完全相同的134个事件、47家公司样本上比较两种逐事件扩展训练的
误差修正模型：

| 模型 | 信息 | 毛利率MAE | 相对基线变化 |
|---|---|---:|---:|
| E0 | 现有财务预测 | 0.01834 | — |
| E1 | E0加公告前市场修正 | 0.02057 | 恶化12.16% |
| E2 | E0加市场及滞后财务修正 | 0.02065 | 恶化12.60% |

每个修正预测只能训练于实际毛利率在当前特征截止时间之前已经可得的事件。标准化只使用
严格更早的样本外预测误差。

### 结果与研究闸门

两个市场增强候选都没有通过“MAE至少相对改善1%”的门槛，因此仍然选择E0。财务预测闸门
失败后，系统不会打开被拒绝模型的收益诊断。

这个结果说明，简单的公告前价格、波动率、成交量、beta和回撤信息，没有改善本样本中的
毛利率预期。它并不意味着市场状态不能预测收益；它只意味着我们不能把这组特征包装成
“更好的会计Innovation”。

下一批预期输入应该更接近会计预期本身：历史公司指引、严格点时的分析师一致预期，以及
结构化的滞后经营变量。每个特征族都应单独加入，并且只有先通过财务预测闸门，才能检查
它与公告后收益的关系。
