# OB to TEAD ticker-transition repair v1

The preserved point-in-time identity evidence records an OB to TEAD ticker
change on 2025-06-10 while CIK, exchange, provider security id, share-class FIGI,
and composite FIGI remain unchanged. The repair collects TEAD unadjusted bars,
splits, dividends, and endpoint identity from the transition date through the
frozen issuer window end.

The reconciliation requires consecutive exchange sessions, non-overlapping old
and new symbol bars, matching endpoint security ids, one boundary ticker event,
no later ticker change, complete corporate-action queries, and an explicit broad
price-scale check. It preserves all original response hashes and does not inspect
or change filing-signal values.
