# Policy-aware shadow order ledger v2

V2 consumes the immutable shadow trading policy in addition to the prospective
event state machine. It retains one decision per frozen event and now exposes
the policy version, signal action, target paper weight, timing rule, and capital
limits. Policy scope must match the state-machine event set exactly.

For a clean future `innovation_ready` event, v2 maps the signal as follows:

- `>= +1%`: `shadow_buy_intent_ready`, target weight 2%;
- `<= -1%`: `shadow_exit_or_avoid_intent_ready`, target weight zero and no short;
- inside the band: `signal_dead_zone`.

An intention still is not an order. The order table stays empty until official
calendar, point-in-time price/cost, current paper portfolio, cash, and risk-cap
evidence are available. Quantity therefore remains zero, broker submission is
false, and live authorization is false.

The current 37-event snapshot contains no Innovation. It records 33
`awaiting_innovation`, two `quality_review_required`, and two
`quality_quarantined` decisions. The trading-policy gate now passes, while the
paper-order gate correctly remains closed.

## 中文摘要

V2 在事件状态机之外正式读取不可变的影子交易政策，并记录政策版本、信号动作、目标模拟权重、
执行规则和资金限制。未来只有干净且达到 `innovation_ready` 的事件才能被映射为买入、退出/回避
或噪声区意图；意图不等于订单。官方日历、点时价格与成本、模拟组合和现金约束没有验证前，订单
表仍为空、数量仍为零，也绝不连接券商或授权实盘。
