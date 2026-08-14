# Preliminary effectiveness pre-registration v1

This bundle freezes the first externally shareable Level/Delta/Innovation
effectiveness experiment before the 15-issuer longitudinal market batch is
collected.

## Frozen primary question

On the exact same point-in-time complete-case filing events, does standardized
out-of-sample gross-margin Innovation explain 20-session post-filing market
excess total returns better than year-over-year Delta?

The primary contrast is the issuer-cluster-bootstrap difference between the
Spearman IC of Innovation and the Spearman IC of Delta. A supporting rank
regression tests whether Innovation remains positive after Delta is included.
Level is retained as a descriptive same-sample reference.

## Outcome timing

- Event time is the SEC filing acceptance timestamp.
- Entry is the second future regular-session open.
- The primary exit is the close of the twentieth holding session.
- One- and five-session outcomes are frozen as secondary sensitivities.
- SPY point-in-time total return is the primary benchmark.
- Sector excess return remains secondary until historical sector mapping passes.

## Verdict boundary

The report cannot claim preliminary supportive evidence unless data-quality and
coverage gates pass, the Innovation IC and its incremental contrast are positive
with issuer-cluster uncertainty excluding zero, and the sign survives the
precommitted issuer-influence and ASNS-tail diagnostics. Otherwise the report
must use the applicable non-supportive verdict stored in `verdict_rules.parquet`.

This experiment does not estimate annualized performance, Sharpe ratio, strategy
PnL, or a buy/sell rule. It is an educational filing-signal validity study.
