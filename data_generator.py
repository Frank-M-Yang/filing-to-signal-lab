"""Deterministic point-in-time financial and market-data simulation.

The simulator is intentionally transparent: named parameters control the weak
economic signal and the three mechanisms that make biased backtests look better.
Nothing in this module represents an estimate from real markets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for one reproducible research dataset."""

    seed: int = 42
    n_stocks: int = 50
    n_quarters: int = 36
    first_report_date: str = "2017-03-31"
    delisted_fraction: float = 0.18
    default_holding_days: int = 20
    maximum_holding_days: int = 45
    # Weak relationship available to an honest backtest.
    post_announcement_alpha_total: float = 0.0015
    # Reaction between announcement close and the second trading-day open.
    announcement_reaction_total: float = 0.036
    # Price pressure attached to misleading final reports of eventual delistings.
    distress_return_penalty: float = -0.080
    # Fraction and future-return loading of revisions used to demonstrate leakage.
    restatement_probability: float = 0.36
    restatement_future_loading: float = 1.20


@dataclass(frozen=True)
class ResearchData:
    """Canonical input expected by the signal and backtest layers."""

    fundamentals: pd.DataFrame
    prices: pd.DataFrame
    benchmark: pd.DataFrame
    metadata: dict[str, str | int | float]


class ResearchDataSource(Protocol):
    """Adapter boundary for simulations or future real point-in-time data."""

    def load(self) -> ResearchData:
        """Return validated fundamentals, stock bars, and benchmark bars."""
        ...


class SimulatedDataSource:
    """Generate the built-in deterministic bias-demonstration dataset."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    def load(self) -> ResearchData:
        """Generate and validate a complete simulated dataset."""
        data = generate_simulated_data(self.config)
        validate_research_data(data)
        return data


def generate_simulated_data(config: SimulationConfig | None = None) -> ResearchData:
    """Generate financial reports, restatements, delistings, and daily prices.

    The generation order matters. Original reports are created first. Prices are
    then simulated using only a weak original-data signal. Finally, a subset of
    restatements is made weakly correlated with already-realized future returns.
    Using those restatements historically is therefore explicit look-ahead bias.
    """
    cfg = config or SimulationConfig()
    _validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    tickers = np.array([f"QF{i:03d}" for i in range(1, cfg.n_stocks + 1)])
    report_dates = pd.date_range(
        cfg.first_report_date, periods=cfg.n_quarters, freq=pd.offsets.QuarterEnd()
    )

    company_index = np.repeat(np.arange(cfg.n_stocks), cfg.n_quarters)
    quarter_index = np.tile(np.arange(cfg.n_quarters), cfg.n_stocks)
    frame = pd.DataFrame(
        {
            "ticker": tickers[company_index],
            "company_index": company_index,
            "quarter_index": quarter_index,
            "report_date": report_dates.take(quarter_index),
        }
    )

    n_delisted = max(1, round(cfg.n_stocks * cfg.delisted_fraction))
    delisted_companies = rng.choice(cfg.n_stocks, size=n_delisted, replace=False)
    delist_quarters = np.full(cfg.n_stocks, cfg.n_quarters + 1, dtype=int)
    delist_quarters[delisted_companies] = rng.integers(
        max(16, cfg.n_quarters - 9), cfg.n_quarters - 1, size=n_delisted
    )
    frame["is_listed"] = ~frame["company_index"].isin(delisted_companies)
    frame["delist_quarter"] = delist_quarters[frame["company_index"].to_numpy()]
    frame = frame.loc[frame["quarter_index"] <= frame["delist_quarter"]].copy()

    company = frame["company_index"].to_numpy(dtype=int)
    quarter = frame["quarter_index"].to_numpy(dtype=int)
    seasonal = np.sin(2.0 * np.pi * quarter / 4.0)
    base_eps = rng.uniform(0.25, 2.4, cfg.n_stocks)
    base_revenue = rng.uniform(700.0, 12_000.0, cfg.n_stocks)
    base_margin = rng.uniform(0.24, 0.64, cfg.n_stocks)
    growth = rng.normal(0.012, 0.006, cfg.n_stocks)

    eps = base_eps[company] * (1.0 + growth[company] * quarter + 0.08 * seasonal) + rng.normal(
        0.0, 0.13, len(frame)
    )
    revenue = (
        base_revenue[company]
        * (1.0 + 0.008 * quarter + 0.025 * seasonal)
        * np.exp(rng.normal(0.0, 0.035, len(frame)))
    )
    gross_margin = (
        base_margin[company]
        + 0.006 * seasonal
        + 0.00035 * quarter
        + rng.normal(0.0, 0.012, len(frame))
    )
    distress_phase = np.clip(
        quarter - (frame["delist_quarter"].to_numpy(dtype=int) - 2), 0, 2
    ).astype(float)
    distress_phase *= (~frame["is_listed"].to_numpy(dtype=bool)).astype(float)
    # The final reports of failed companies look deceptively strong before prices fall.
    eps += distress_phase * base_eps[company] * 1.20
    revenue *= 1.0 + distress_phase * 0.060
    gross_margin += distress_phase * 0.030

    frame["revenue"] = np.maximum(revenue, 1.0)
    frame["eps"] = eps
    frame["gross_margin"] = np.clip(gross_margin, 0.05, 0.85)
    announcement_offsets = 26 + (company % 5) + rng.integers(0, 5, len(frame))
    frame["announce_date"] = [
        report_date + pd.offsets.BusinessDay(int(offset))
        for report_date, offset in zip(frame["report_date"], announcement_offsets, strict=True)
    ]
    frame["announce_date"] = pd.to_datetime(frame["announce_date"])

    frame = _add_original_signal_components(frame)
    sue_z = _cross_sectional_zscore(frame["sue_original"], frame["report_date"])
    margin_z = _cross_sectional_zscore(frame["gross_margin_change_original"], frame["report_date"])
    frame["latent_quality"] = (0.65 * sue_z + 0.35 * margin_z).fillna(0.0).clip(-2.5, 2.5)
    frame["distress_phase"] = distress_phase

    earliest_date = frame["announce_date"].min() - pd.offsets.BusinessDay(5)
    latest_date = frame["announce_date"].max() + pd.offsets.BusinessDay(
        cfg.maximum_holding_days + 8
    )
    trading_dates = pd.bdate_range(earliest_date, latest_date)
    date_positions = pd.Series(np.arange(len(trading_dates)), index=trading_dates)
    announcement_positions = frame["announce_date"].map(date_positions).to_numpy(dtype=int)
    company_positions = frame["company_index"].to_numpy(dtype=int)
    quality = frame["latent_quality"].to_numpy(dtype=float)

    market_log_return = 0.00020 + rng.normal(0.0, 0.0075, len(trading_dates))
    betas = rng.normal(1.0, 0.14, cfg.n_stocks)
    idiosyncratic = rng.normal(0.0, 0.0105, (len(trading_dates), cfg.n_stocks))
    stock_total = market_log_return[:, None] * betas[None, :] + idiosyncratic
    overnight = 0.30 * stock_total + rng.normal(0.0, 0.0018, (len(trading_dates), cfg.n_stocks))
    intraday = stock_total - overnight

    reaction_each_day = cfg.announcement_reaction_total / 2.0
    for offset in (1, 2):
        np.add.at(
            overnight,
            (announcement_positions + offset, company_positions),
            reaction_each_day * quality,
        )

    daily_alpha = cfg.post_announcement_alpha_total / cfg.default_holding_days
    daily_distress = cfg.distress_return_penalty / cfg.default_holding_days
    distress = frame["distress_phase"].to_numpy(dtype=float)
    corrected_entries = announcement_positions + 2
    for offset in range(cfg.default_holding_days):
        np.add.at(
            intraday,
            (corrected_entries + offset, company_positions),
            daily_alpha * quality + daily_distress * distress,
        )

    benchmark_overnight = 0.30 * market_log_return + rng.normal(0.0, 0.0008, len(trading_dates))
    benchmark_intraday = market_log_return - benchmark_overnight
    stock_open, stock_close = _prices_from_log_returns(overnight, intraday, rng)
    benchmark_open, benchmark_close = _prices_from_log_returns(
        benchmark_overnight[:, None], benchmark_intraday[:, None], rng
    )

    corrected_excess = _event_excess_returns(
        stock_open,
        stock_close,
        benchmark_open[:, 0],
        benchmark_close[:, 0],
        corrected_entries,
        company_positions,
        cfg.default_holding_days,
    )
    future_z = _cross_sectional_zscore(
        pd.Series(corrected_excess, index=frame.index), frame["report_date"]
    ).fillna(0.0)
    eligible = frame["sue_original"].notna().to_numpy()
    has_restatement = (rng.random(len(frame)) < cfg.restatement_probability) & eligible
    revision_noise = rng.normal(0.0, 0.10, len(frame))
    eps_scale = frame["eps_history_std"].fillna(frame["eps"].abs() * 0.15 + 0.05)
    revision_driver = (
        cfg.restatement_future_loading * future_z.to_numpy() + revision_noise
    ) * has_restatement
    frame["eps_restated"] = frame["eps"] + eps_scale * revision_driver
    frame["revenue_restated"] = frame["revenue"] * (
        1.0 + has_restatement * (0.0055 * future_z.to_numpy() + rng.normal(0.0, 0.0015, len(frame)))
    )
    frame["gross_margin_restated"] = np.clip(
        frame["gross_margin"]
        + has_restatement * (0.010 * future_z.to_numpy() + rng.normal(0.0, 0.0025, len(frame))),
        0.05,
        0.85,
    )
    frame["has_restatement"] = has_restatement
    restatement_lags = rng.integers(40, 141, len(frame))
    frame["restatement_date"] = pd.NaT
    frame.loc[has_restatement, "restatement_date"] = [
        date + pd.offsets.BusinessDay(int(lag))
        for date, lag in zip(
            frame.loc[has_restatement, "announce_date"],
            restatement_lags[has_restatement],
            strict=True,
        )
    ]

    final_announcement = frame.groupby("company_index")["announce_date"].max()
    delist_dates: dict[int, pd.Timestamp] = {}
    for company_id in delisted_companies:
        delist_dates[int(company_id)] = final_announcement.loc[
            int(company_id)
        ] + pd.offsets.BusinessDay(cfg.maximum_holding_days + 3)
    frame["delist_date"] = frame["company_index"].map(delist_dates)

    prices = pd.DataFrame(
        {
            "date": np.repeat(trading_dates.to_numpy(), cfg.n_stocks),
            "ticker": np.tile(tickers, len(trading_dates)),
            "open": stock_open.ravel(),
            "close": stock_close.ravel(),
        }
    )
    ticker_delist_dates = {tickers[key]: value for key, value in delist_dates.items()}
    mapped_delist = prices["ticker"].map(ticker_delist_dates)
    prices = prices.loc[mapped_delist.isna() | (prices["date"] <= mapped_delist)].reset_index(
        drop=True
    )
    benchmark = pd.DataFrame(
        {
            "date": trading_dates,
            "open": benchmark_open[:, 0],
            "close": benchmark_close[:, 0],
        }
    )

    public_columns = [
        "ticker",
        "report_date",
        "announce_date",
        "revenue",
        "eps",
        "gross_margin",
        "revenue_restated",
        "eps_restated",
        "gross_margin_restated",
        "has_restatement",
        "restatement_date",
        "is_listed",
        "delist_date",
    ]
    fundamentals = (
        frame[public_columns].sort_values(["ticker", "report_date"]).reset_index(drop=True)
    )
    metadata: dict[str, str | int | float] = {
        "source": "deterministic_simulation",
        "seed": cfg.seed,
        "n_stocks": cfg.n_stocks,
        "n_quarters": cfg.n_quarters,
        "delisted_stocks": n_delisted,
        "calibration": "bias-demonstration-not-empirical",
    }
    return ResearchData(fundamentals, prices, benchmark, metadata)


def validate_research_data(data: ResearchData) -> None:
    """Fail fast when an adapter does not honor the canonical data contract."""
    fundamental_columns = {
        "ticker",
        "report_date",
        "announce_date",
        "revenue",
        "eps",
        "gross_margin",
        "revenue_restated",
        "eps_restated",
        "gross_margin_restated",
        "is_listed",
    }
    price_columns = {"date", "ticker", "open", "close"}
    benchmark_columns = {"date", "open", "close"}
    missing_fundamentals = fundamental_columns.difference(data.fundamentals.columns)
    missing_prices = price_columns.difference(data.prices.columns)
    missing_benchmark = benchmark_columns.difference(data.benchmark.columns)
    if missing_fundamentals or missing_prices or missing_benchmark:
        raise ValueError(
            "Data-source contract violation: "
            f"fundamentals={sorted(missing_fundamentals)}, "
            f"prices={sorted(missing_prices)}, benchmark={sorted(missing_benchmark)}"
        )
    if data.fundamentals.duplicated(["ticker", "report_date"]).any():
        raise ValueError("Fundamental rows must be unique by ticker and report_date")
    if data.prices.duplicated(["ticker", "date"]).any():
        raise ValueError("Price rows must be unique by ticker and date")
    if (data.fundamentals["announce_date"] <= data.fundamentals["report_date"]).any():
        raise ValueError("Every announcement must occur after its report period")
    if (data.prices[["open", "close"]] <= 0.0).any().any():
        raise ValueError("All simulated prices must be strictly positive")


def _validate_config(config: SimulationConfig) -> None:
    if config.n_stocks < 10:
        raise ValueError("At least ten stocks are required for quintile portfolios")
    if config.n_quarters < 16:
        raise ValueError("At least sixteen quarters are required for SUE warm-up")
    if not 0.0 < config.delisted_fraction < 0.5:
        raise ValueError("delisted_fraction must be between zero and 0.5")
    if config.maximum_holding_days < config.default_holding_days:
        raise ValueError("maximum_holding_days cannot be below default_holding_days")


def _add_original_signal_components(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["ticker", "report_date"]).copy()
    eps_group = ordered.groupby("ticker", sort=False)["eps"]
    margin_group = ordered.groupby("ticker", sort=False)["gross_margin"]
    expected_eps = eps_group.shift(4)
    eps_std = eps_group.transform(
        lambda values: values.shift(1).rolling(8, min_periods=8).std(ddof=1)
    )
    safe_std = eps_std.where(eps_std.abs() > 1e-10)
    ordered["eps_history_std"] = eps_std
    ordered["sue_original"] = (ordered["eps"] - expected_eps) / safe_std
    ordered["gross_margin_change_original"] = ordered["gross_margin"] - margin_group.shift(4)
    return ordered.sort_index()


def _cross_sectional_zscore(values: pd.Series, groups: pd.Series) -> pd.Series:
    means = values.groupby(groups).transform("mean")
    standard_deviations = values.groupby(groups).transform("std")
    return (values - means) / standard_deviations.where(standard_deviations > 1e-12)


def _prices_from_log_returns(
    overnight: NDArray[np.float64],
    intraday: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    start: NDArray[np.float64] = np.asarray(
        rng.uniform(25.0, 120.0, size=overnight.shape[1]), dtype=np.float64
    )
    total = overnight + intraday
    prior_cumulative = np.vstack([np.zeros((1, total.shape[1])), np.cumsum(total, axis=0)[:-1]])
    open_prices = start[None, :] * np.exp(prior_cumulative + overnight)
    close_prices = open_prices * np.exp(intraday)
    return open_prices, close_prices


def _event_excess_returns(
    stock_open: NDArray[np.float64],
    stock_close: NDArray[np.float64],
    benchmark_open: NDArray[np.float64],
    benchmark_close: NDArray[np.float64],
    entry_positions: NDArray[np.int64],
    company_positions: NDArray[np.int64],
    holding_days: int,
) -> NDArray[np.float64]:
    exit_positions = entry_positions + holding_days
    stock_returns = (
        stock_close[exit_positions, company_positions]
        / stock_open[entry_positions, company_positions]
        - 1.0
    )
    benchmark_returns = benchmark_close[exit_positions] / benchmark_open[entry_positions] - 1.0
    return stock_returns - benchmark_returns
