"""Phase-0 cost and liquidity gate for the small-cap execution universe.

This module is the referee that decides, *before any signal research*, which
candidate names are even tradable and how large a gross edge a signal must
clear to survive execution costs.  It is pure computation: it downloads no
data, needs no credentials, and reads no return outcomes.

Control-theory framing (see docs/small_cap_pead_pivot_v1.md, section 5.1):
execution is a control input and cost is the plant's cost response.  The
liquidity floor is a *controllability boundary* -- a name is controllable only
if a reference position can be entered and exited without cost consuming the
edge.  While the traded notional stays far below available dollar volume, the
market-impact term is negligible, which is the principled reason this first
version models only the spread, commission, and slippage terms and defers
dynamic optimal execution (Almgren-Chriss) to a later, larger-size regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

COST_GATE_VERSION: Final[str] = "small-cap-cost-gate-v1"

CANDIDATE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "candidate_id",
    "price",
    "relative_spread",
    "median_dollar_volume_usd",
)

BASIS_POINTS: Final[float] = 10_000.0


class CostGateError(ValueError):
    """Raised when cost-gate inputs or parameters violate the frozen design."""


@dataclass(frozen=True)
class CostGateConfig:
    """Explicit, outcome-blind execution-cost and controllability assumptions.

    ``max_participation`` is the fraction of median dollar volume a single
    reference position may consume.  Staying under it keeps market impact
    negligible, which is why only the fixed spread/commission/slippage terms
    are modelled here.
    """

    position_notional_usd: float = 10_000.0
    max_participation: float = 0.01
    min_price_usd: float = 5.0
    max_relative_spread: float = 0.02
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    version: str = COST_GATE_VERSION

    @property
    def minimum_median_dollar_volume_usd(self) -> float:
        """Dollar-volume floor implied by the controllability (participation) cap."""
        return self.position_notional_usd / self.max_participation

    def validate(self) -> None:
        """Reject ambiguous or economically invalid cost parameters."""
        if self.position_notional_usd <= 0:
            raise CostGateError("position_notional_usd must be positive")
        if not 0 < self.max_participation <= 0.10:
            raise CostGateError("max_participation must lie in (0, 0.10]")
        if self.min_price_usd < 0:
            raise CostGateError("min_price_usd cannot be negative")
        if not 0 < self.max_relative_spread < 1:
            raise CostGateError("max_relative_spread must lie in (0, 1)")
        if self.commission_bps < 0:
            raise CostGateError("commission_bps cannot be negative")
        if self.slippage_bps < 0:
            raise CostGateError("slippage_bps cannot be negative")
        if not self.version.strip():
            raise CostGateError("version cannot be blank")


def evaluate_cost_gate(
    candidates: pd.DataFrame,
    config: CostGateConfig | None = None,
) -> pd.DataFrame:
    """Score each candidate for tradability and round-trip execution cost.

    The returned frame keeps the input columns and adds:

    - ``one_way_cost_bps`` / ``round_trip_cost_bps``: enter-only and
      enter-plus-exit cost as a fraction of notional, in basis points.  A
      hold-then-exit strategy pays one round trip.
    - ``net_edge_hurdle_bps``: the minimum gross edge a signal must beat to be
      net-positive.  It equals the round-trip cost.
    - ``participation``: reference notional divided by median dollar volume.
    - ``controllable``: whether participation stays within the cap.
    - ``tradable`` and ``exclusion_reason``: the liquidity-floor verdict, where
      the reason names the first failing floor (price, spread, then liquidity).
    """
    cfg = config or CostGateConfig()
    cfg.validate()
    _require_columns(candidates, CANDIDATE_REQUIRED_COLUMNS)
    result = candidates.copy()
    if result["candidate_id"].isna().any() or result["candidate_id"].duplicated().any():
        raise CostGateError("candidate_id must be present and unique")

    price = _finite_numeric(result["price"], "price")
    relative_spread = _finite_numeric(result["relative_spread"], "relative_spread")
    dollar_volume = _finite_numeric(result["median_dollar_volume_usd"], "median_dollar_volume_usd")
    if (price <= 0).any():
        raise CostGateError("price must be positive")
    if (relative_spread < 0).any():
        raise CostGateError("relative_spread cannot be negative")
    if (dollar_volume <= 0).any():
        raise CostGateError("median_dollar_volume_usd must be positive")

    one_way_fraction = (
        relative_spread / 2.0 + cfg.commission_bps / BASIS_POINTS + cfg.slippage_bps / BASIS_POINTS
    )
    round_trip_fraction = 2.0 * one_way_fraction
    participation = cfg.position_notional_usd / dollar_volume

    below_min_price = price < cfg.min_price_usd
    wide_spread = relative_spread > cfg.max_relative_spread
    uncontrollable = participation > cfg.max_participation

    exclusion_reason = np.select(
        [
            below_min_price.to_numpy(dtype=bool),
            wide_spread.to_numpy(dtype=bool),
            uncontrollable.to_numpy(dtype=bool),
        ],
        ["excluded_min_price", "excluded_wide_spread", "excluded_illiquid_uncontrollable"],
        default="tradable",
    )

    result["one_way_cost_bps"] = (one_way_fraction * BASIS_POINTS).to_numpy(dtype=float)
    result["round_trip_cost_bps"] = (round_trip_fraction * BASIS_POINTS).to_numpy(dtype=float)
    result["net_edge_hurdle_bps"] = result["round_trip_cost_bps"]
    result["participation"] = participation.to_numpy(dtype=float)
    result["controllable"] = (~uncontrollable).to_numpy(dtype=bool)
    result["exclusion_reason"] = exclusion_reason
    result["tradable"] = result["exclusion_reason"] == "tradable"
    return result.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)


def apply_net_edge_hurdle(
    evaluated: pd.DataFrame,
    *,
    gross_edge_bps_column: str = "gross_edge_bps",
) -> pd.DataFrame:
    """Compare a signal's gross edge against each name's round-trip cost.

    ``net_edge_bps = gross_edge_bps - round_trip_cost_bps``.  A name
    ``clears_hurdle`` only when it is tradable *and* its net edge is positive:
    a gross edge earned inside an untradable name can never be banked.
    """
    _require_columns(evaluated, (gross_edge_bps_column, "round_trip_cost_bps", "tradable"))
    result = evaluated.copy()
    gross = _finite_numeric(result[gross_edge_bps_column], gross_edge_bps_column)
    round_trip = _finite_numeric(result["round_trip_cost_bps"], "round_trip_cost_bps")
    net_edge = (gross - round_trip).to_numpy(dtype=float)
    result["net_edge_bps"] = net_edge
    result["clears_hurdle"] = result["tradable"].to_numpy(dtype=bool) & (net_edge > 0.0)
    return result


def summarize_cost_gate(evaluated: pd.DataFrame) -> dict[str, float | int]:
    """Aggregate a scored cohort into headline tradability and cost figures."""
    _require_columns(evaluated, ("tradable", "round_trip_cost_bps", "net_edge_hurdle_bps"))
    tradable = evaluated["tradable"].astype(bool)
    tradable_frame = evaluated.loc[tradable]
    empty = tradable_frame.empty
    return {
        "candidates": len(evaluated),
        "tradable": int(tradable.sum()),
        "excluded": int((~tradable).sum()),
        "median_round_trip_cost_bps": (
            float("nan") if empty else float(tradable_frame["round_trip_cost_bps"].median())
        ),
        "median_net_edge_hurdle_bps": (
            float("nan") if empty else float(tradable_frame["net_edge_hurdle_bps"].median())
        ),
    }


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CostGateError(f"Missing cost-gate columns: {sorted(missing)}")


def _finite_numeric(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise CostGateError(f"{name} contains a non-finite value")
    return numeric
