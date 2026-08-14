"""Event-driven long-short backtest with explicit bias switches."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from data_generator import ResearchData
from signals import SignalName, calculate_signal


@dataclass(frozen=True)
class BiasConfig:
    """Deliberately introduce common historical-data biases when enabled."""

    lookahead: bool = False
    survivorship: bool = False
    announcement_timing: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    """Transparent event-study and portfolio assumptions."""

    holding_days: int = 20
    quantile_fraction: float = 0.20

    def __post_init__(self) -> None:
        if not 1 <= self.holding_days <= 45:
            raise ValueError("holding_days must be between 1 and 45")
        if not 0.10 <= self.quantile_fraction <= 0.40:
            raise ValueError("quantile_fraction must be between 0.10 and 0.40")


@dataclass(frozen=True)
class BacktestMetrics:
    """Headline statistics for one scenario."""

    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    mean_ic: float
    hit_rate: float
    observations: int
    cohorts: int


@dataclass(frozen=True)
class BacktestResult:
    """Metrics and audit tables produced by one backtest."""

    metrics: BacktestMetrics
    equity_curve: pd.DataFrame
    cohort_returns: pd.DataFrame
    quantile_returns: pd.DataFrame
    events: pd.DataFrame
    biases: BiasConfig
    config: BacktestConfig
    signal_name: SignalName


STANDARD_SCENARIOS: dict[str, BiasConfig] = {
    "原始回测（全部偏差）": BiasConfig(True, True, True),
    "修正前视偏差后": BiasConfig(False, True, True),
    "修正全部偏差后": BiasConfig(False, False, False),
}


def run_backtest(
    data: ResearchData,
    signal_name: SignalName,
    config: BacktestConfig | None = None,
    biases: BiasConfig | None = None,
) -> BacktestResult:
    """Run one event-driven, gross-100%, dollar-neutral quantile portfolio.

    Honest execution uses the second trading-day open after an announcement.
    When announcement-timing bias is enabled, the impossible announcement-close
    price is used instead. All future-return calculations are performed only after
    the signal has been formed.
    """
    cfg = config or BacktestConfig()
    bias = biases or BiasConfig()
    events = calculate_signal(data.fundamentals, signal_name, use_restated=bias.lookahead).dropna(
        subset=["signal"]
    )
    if bias.survivorship:
        events = events.loc[events["is_listed"]].copy()
    if events.empty:
        raise ValueError("No valid signal observations remain after applying filters")

    trading_dates = pd.DatetimeIndex(data.benchmark["date"].sort_values().unique())
    announcement_dates = pd.DatetimeIndex(pd.to_datetime(events["announce_date"]))
    announcement_positions = trading_dates.get_indexer(announcement_dates)
    if (announcement_positions < 0).any():
        raise ValueError("Every announcement date must exist in the benchmark calendar")
    entry_positions = (
        announcement_positions if bias.announcement_timing else announcement_positions + 2
    )
    exit_positions = entry_positions + cfg.holding_days
    valid_horizon = exit_positions < len(trading_dates)
    events = events.loc[valid_horizon].copy()
    entry_positions = entry_positions[valid_horizon]
    exit_positions = exit_positions[valid_horizon]
    if events.empty:
        raise ValueError("No event has enough price history for the requested holding period")

    events["entry_date"] = trading_dates.take(entry_positions).to_numpy()
    events["exit_date"] = trading_dates.take(exit_positions).to_numpy()
    entry_field = "close" if bias.announcement_timing else "open"
    price_table = data.prices.set_index(["date", "ticker"]).sort_index()
    entry_keys = pd.MultiIndex.from_arrays(
        [events["entry_date"], events["ticker"]], names=["date", "ticker"]
    )
    exit_keys = pd.MultiIndex.from_arrays(
        [events["exit_date"], events["ticker"]], names=["date", "ticker"]
    )
    events["entry_price"] = price_table[entry_field].reindex(entry_keys).to_numpy()
    events["exit_price"] = price_table["close"].reindex(exit_keys).to_numpy()
    benchmark = data.benchmark.set_index("date").sort_index()
    events["benchmark_entry"] = benchmark[entry_field].reindex(events["entry_date"]).to_numpy()
    events["benchmark_exit"] = benchmark["close"].reindex(events["exit_date"]).to_numpy()
    events = events.dropna(
        subset=["entry_price", "exit_price", "benchmark_entry", "benchmark_exit"]
    ).copy()
    if events.empty:
        raise ValueError("No event has complete executable prices")

    events["stock_return"] = events["exit_price"] / events["entry_price"] - 1.0
    events["benchmark_return"] = events["benchmark_exit"] / events["benchmark_entry"] - 1.0
    events["excess_return"] = events["stock_return"] - events["benchmark_return"]
    events["rank_percentile"] = events.groupby("report_date")["signal"].rank(
        pct=True, method="first"
    )
    events["quantile"] = np.ceil(events["rank_percentile"] * 5.0).clip(1, 5).astype(int)

    cohort_rows: list[dict[str, object]] = []
    ic_rows: list[tuple[pd.Timestamp, float]] = []
    threshold = cfg.quantile_fraction
    for report_date, cohort in events.groupby("report_date", sort=True):
        report_timestamp = pd.Timestamp(str(report_date))
        long_leg = cohort.loc[cohort["rank_percentile"] > 1.0 - threshold]
        short_leg = cohort.loc[cohort["rank_percentile"] <= threshold]
        if long_leg.empty or short_leg.empty:
            continue
        long_return = float(long_leg["excess_return"].mean())
        short_return = float(short_leg["excess_return"].mean())
        strategy_return = 0.5 * (long_return - short_return)
        cohort_rows.append(
            {
                "report_date": report_timestamp,
                "return_date": pd.Timestamp(cohort["exit_date"].max()),
                "strategy_return": strategy_return,
                "long_return": long_return,
                "short_return": short_return,
                "long_count": len(long_leg),
                "short_count": len(short_leg),
            }
        )
        ic_rows.append((report_timestamp, _spearman(cohort["signal"], cohort["excess_return"])))
    cohort_returns = pd.DataFrame(cohort_rows).sort_values("return_date").reset_index(drop=True)
    if cohort_returns.empty:
        raise ValueError("No report cohort contains both long and short positions")
    ic_frame = pd.DataFrame(ic_rows, columns=["report_date", "ic"])
    cohort_returns = cohort_returns.merge(ic_frame, on="report_date", how="left")

    cumulative = (1.0 + cohort_returns["strategy_return"]).cumprod()
    initial_date = pd.Timestamp(events["entry_date"].min())
    equity_curve = pd.concat(
        [
            pd.DataFrame({"date": [initial_date], "equity": [1.0]}),
            pd.DataFrame(
                {
                    "date": cohort_returns["return_date"].to_numpy(),
                    "equity": cumulative.to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    quantile_returns = (
        events.groupby("quantile", as_index=False)
        .agg(
            mean_excess_return=("excess_return", "mean"),
            observations=("excess_return", "size"),
        )
        .sort_values(by="quantile")
    )
    metrics = _calculate_metrics(cohort_returns, cumulative, events)
    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        cohort_returns=cohort_returns,
        quantile_returns=quantile_returns,
        events=events.reset_index(drop=True),
        biases=bias,
        config=cfg,
        signal_name=signal_name,
    )


def compare_standard_scenarios(
    data: ResearchData,
    signal_name: SignalName,
    config: BacktestConfig | None = None,
) -> dict[str, BacktestResult]:
    """Run the three fixed audit scenarios displayed by the application."""
    cfg = config or BacktestConfig()
    return {
        name: run_backtest(data, signal_name, cfg, biases)
        for name, biases in STANDARD_SCENARIOS.items()
    }


def calculate_bias_impact(
    data: ResearchData,
    signal_name: SignalName,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Measure the change from removing each bias independently and all together."""
    cfg = config or BacktestConfig()
    scenarios = {
        "带全部偏差": BiasConfig(True, True, True),
        "仅修正前视偏差": BiasConfig(False, True, True),
        "仅修正幸存者偏差": BiasConfig(True, False, True),
        "仅修正公告延迟偏差": BiasConfig(True, True, False),
        "修正全部偏差": BiasConfig(False, False, False),
    }
    results = {name: run_backtest(data, signal_name, cfg, bias) for name, bias in scenarios.items()}
    baseline = results["带全部偏差"].metrics.annual_return
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "情景": name,
                "年化收益": result.metrics.annual_return,
                "相对带偏差基准的收益削弱": baseline - result.metrics.annual_return,
                "夏普比率": result.metrics.sharpe_ratio,
                "IC": result.metrics.mean_ic,
            }
        )
    return pd.DataFrame(rows)


def _calculate_metrics(
    cohort_returns: pd.DataFrame,
    cumulative: pd.Series,
    events: pd.DataFrame,
) -> BacktestMetrics:
    returns = cohort_returns["strategy_return"].astype(float)
    periods = len(returns)
    ending_equity = float(cumulative.iloc[-1])
    annual_return = ending_equity ** (4.0 / periods) - 1.0 if ending_equity > 0.0 else -1.0
    volatility = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / volatility * sqrt(4.0)) if volatility > 1e-12 else 0.0
    drawdown = cumulative / cumulative.cummax() - 1.0
    return BacktestMetrics(
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        max_drawdown=float(drawdown.min()),
        mean_ic=float(cohort_returns["ic"].mean()),
        hit_rate=float((returns > 0.0).mean()),
        observations=len(events),
        cohorts=periods,
    )


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3:
        return float("nan")
    left_rank = valid.iloc[:, 0].rank(method="average")
    right_rank = valid.iloc[:, 1].rank(method="average")
    correlation = left_rank.corr(right_rank)
    return float(correlation) if pd.notna(correlation) else float("nan")
