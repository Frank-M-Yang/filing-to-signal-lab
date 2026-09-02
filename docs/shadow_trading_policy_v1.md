# Shadow trading policy v1

## English

### Purpose and timing

This protocol was frozen on 2026-08-31 while all 37 registered forward events
still lacked their target SEC filing, actual revenue, Innovation, and
post-filing return. It converts the already-preregistered Raw Revenue Market
Innovation into a future paper intention. It does not fit a model, read an
outcome, submit an order, or authorize live trading.

The 1% materiality band and capital limits are ex-ante engineering choices, not
empirically validated optimal parameters. Their value is that they can now be
tested honestly without changing them after outcomes arrive.

### Frozen signal actions

| Raw Revenue Innovation | Shadow action |
| --- | --- |
| `>= +1%` | Create a cash-only buy intention. |
| `<= -1%` | Exit an existing long or remain in cash; never open a short. |
| `(-1%, +1%)` | No trade; treat the observation as inside the materiality band. |

The primary score remains `actual revenue / last eligible pre-SEC-acceptance
consensus revenue - 1`. Quality-review, quarantined, rejected, or non-forward
events cannot enter the policy.

### Capital and execution rules

- target weight per accepted event: 2% of paper NAV;
- maximum weight per issuer: 2%;
- maximum gross invested weight: 20%;
- maximum concurrent positions: 10;
- cash-only long portfolio; short selling and leverage are forbidden;
- entry: unadjusted open of the second official trading session strictly after
  SEC acceptance;
- planned exit: unadjusted close of the twentieth session including the entry
  session;
- round-trip cost: pre-acceptance Corwin–Schultz proxy plus 5 bps, clipped to
  10–100 bps, matching the v3 research protocol;
- missing calendar, price, cost, corporate-action, or portfolio evidence blocks
  the order instead of being imputed.

The policy produces an **intention**, not an executable quantity. A later layer
must prove the official session timestamps, point-in-time cost evidence, paper
NAV, available cash, issuer/concurrency caps, and current holdings before an
order can exist.

### Immutable outputs

- `shadow_trading_policy_events.parquet`: the exact 37-event admitted scope;
- `shadow_trading_policy_rules.parquet`: human-readable frozen rules;
- `shadow_trading_policy_gates.parquet`: outcome and execution firewalls;
- `shadow_trading_policy_protocol.json`: machine-readable protocol;
- report and manifest files with upstream and content hashes.

Rebuild the same immutable policy:

```powershell
uv run python -m scripts.freeze_shadow_trading_policy `
  --prospective-preregistration-root data/processed/prospective_market_innovation_preregistration/<version>
```

## 中文

### 目的与冻结时点

本协议于 2026-08-31 冻结；当时 37 个前瞻事件都还没有目标 SEC 财报、实际收入、Innovation
或公告后收益。它只把已经预注册的 Raw Revenue Market Innovation 映射为未来模拟意图，不拟合
模型、不读取结果、不提交订单，也不授权实盘。

1% 的显著性区间和资金上限是事前工程假设，并不是已经被数据证明最优的参数。它们的价值在于：
未来结果出现后，我们可以诚实检验，而不能再根据结果修改规则。

### 冻结的信号动作

| Raw Revenue Innovation | 影子动作 |
| --- | --- |
| `>= +1%` | 生成仅使用现金的买入意图。 |
| `<= -1%` | 若已有多头则退出，否则保持现金；禁止开空仓。 |
| `(-1%, +1%)` | 不交易，视为处于显著性噪声区间。 |

主信号仍然是“实际收入 ÷ SEC acceptance 前最后一次合格一致预期 − 1”。需要复核、被隔离、
被拒绝或非前瞻事件都不能进入这套政策。

### 资金与执行规则

- 每个合格事件目标权重：模拟净值的 2%；
- 单发行人上限：2%；
- 总投入上限：20%；
- 最多同时持有 10 个仓位；
- 仅现金多头，禁止卖空和杠杆；
- SEC acceptance 后严格第二个官方交易日的未复权开盘价进入；
- 从进入日计数，第 20 个交易日的未复权收盘价退出；
- 往返成本沿用 v3：公告前 Corwin–Schultz 代理加 5bps，并限制在 10–100bps；
- 日历、价格、成本、公司行为或组合状态证据缺失时直接阻塞，不做填充。

协议只生成“意图”，不生成可执行股数。后续模块必须验证官方交易时钟、点时成本、模拟净值、
可用现金、发行人和并发上限以及当前持仓，之后才允许产生影子订单。
