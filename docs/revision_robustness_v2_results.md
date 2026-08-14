# Revision Robustness v2 Results

## Source-audited primary result

Version 2 applies the filing-level tail audit before forming the walk-forward
revision distribution.  Four verified XBRL semantic collisions are restored to
their unchanged consolidated gross margins.  Scope and presentation redraws are
retained because they are genuine future changes to the published historical
signal available to a hindsight data user.

The sample still contains 812 locked M2 events: 724 outcomes are resolved by a
later comparative filing or amendment, and 88 are right-censored.  There are 45
changed resolved outcomes after the source corrections.  The frozen history gate
produces 503 decisions and 417 later-resolved evaluation events.

| Confidence | Median buffer | Empirical coverage | Robust decision share | Robust sign flips |
|---:|---:|---:|---:|---:|
| 80.0% | 0.000 pp | 94.2% | 100.0% | 3 |
| 90.0% | 0.000 pp | 94.2% | 100.0% | 3 |
| 95.0% | 0.166 pp | 97.4% | 90.3% | 3 |
| 97.5% | 4.311 pp | 100.0% | 15.5% | 0 |
| 99.0% | 7.440 pp | 100.0% | 6.0% | 0 |

The valid interpretation is:

> A 0.166 pp pooled buffer is a useful first-pass guard against ordinary
> published-history instability, but it cannot protect every event from later
> scope or presentation changes. Near-complete historical protection removes
> most signals and is not a practical universal decision rule.

The candidate structural-change text gate did not catch any of the three remaining
direction flips and is not deployed.  See
[Revision Tail Source Audit v1](revision_tail_audit_v1.md) for the classifications,
source contract, and negative gate result.

This result measures signal stability only.  It is not evidence of return
predictability, a position-sizing rule, or an investment recommendation.
