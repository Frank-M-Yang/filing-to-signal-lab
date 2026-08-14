# Longitudinal market batch v1

This is the final market-data acquisition phase for the pre-registered
Level/Delta/Innovation effectiveness report.

- One unadjusted daily-bar range is collected for each of 15 issuers.
- Point-in-time identity, splits, and dividends are audited separately.
- The previously identified ISPO ticker-events fallback is constrained to its
  four 20-session event windows instead of the entire issuer-year range.
- SPY unadjusted bars, splits, and dividends cover every frozen 1-, 5-, and
  20-session entry/exit date.
- Existing issuer-level delisting outcomes are reused only after CIK/security
  identity reconciliation.

The collection is resumable and preserves exact response bytes and SHA-256
hashes before parsing. It does not calculate returns or inspect signal outcomes.
