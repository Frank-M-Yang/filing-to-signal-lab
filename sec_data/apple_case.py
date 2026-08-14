"""Curated, auditable Apple 2009 amendment case study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CASE_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "examples"
CASE_FACTS_PATH = CASE_DIRECTORY / "apple_2009_signal_case.csv"
CASE_METADATA_PATH = CASE_DIRECTORY / "apple_2009_case_metadata.json"

METRIC_LABELS = {
    "EarningsPerShareBasic": "基本每股收益",
    "EarningsPerShareDiluted": "稀释每股收益",
    "GrossProfit": "毛利润",
    "NetIncomeLoss": "净利润",
    "SalesRevenueNet": "营业收入",
}

REQUIRED_FACT_COLUMNS = {
    "amendment_fact_id",
    "tag",
    "unit",
    "period_start",
    "period_end",
    "original_accession",
    "original_accepted_at",
    "original_value",
    "amendment_accession",
    "amendment_accepted_at",
    "amendment_value",
    "percentage_difference",
    "comparison_status",
}


@dataclass(frozen=True)
class AppleAmendmentCase:
    """Signal-relevant facts and derived gross-margin history for one amendment."""

    facts: pd.DataFrame
    gross_margin: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def information_delay_days(self) -> float:
        """Elapsed days between the original and amended filing acceptance times."""
        original = pd.Timestamp(self.metadata["original_accepted_at"])
        amendment = pd.Timestamp(self.metadata["amendment_accepted_at"])
        return float((amendment - original).total_seconds() / 86_400)


def load_apple_amendment_case(
    facts_path: Path = CASE_FACTS_PATH,
    metadata_path: Path = CASE_METADATA_PATH,
) -> AppleAmendmentCase:
    """Load and validate the repository's offline Apple amendment case."""
    metadata_document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_document, dict):
        raise ValueError("Apple case metadata must be a JSON object")
    metadata: dict[str, Any] = metadata_document
    facts = pd.read_csv(
        facts_path,
        dtype={
            "amendment_fact_id": "string",
            "original_accession": "string",
            "amendment_accession": "string",
        },
    )
    missing = REQUIRED_FACT_COLUMNS.difference(facts.columns)
    if missing:
        raise ValueError(f"Apple case facts are missing columns: {sorted(missing)}")
    for column in ("original_accepted_at", "amendment_accepted_at"):
        facts[column] = pd.to_datetime(facts[column], utc=True, errors="coerce")
    facts["period_end"] = pd.to_datetime(facts["period_end"], errors="coerce")
    facts["fiscal_year"] = facts["period_end"].dt.year.astype("Int64")
    facts["metric_label"] = facts["tag"].map(METRIC_LABELS)
    _validate_case(facts, metadata)
    gross_margin = _derive_gross_margin(facts)
    return AppleAmendmentCase(
        facts=facts.sort_values(["fiscal_year", "tag"]).reset_index(drop=True),
        gross_margin=gross_margin,
        metadata=metadata,
    )


def _validate_case(facts: pd.DataFrame, metadata: dict[str, Any]) -> None:
    expected_count = int(metadata["curated_fact_count"])
    if len(facts) != expected_count:
        raise ValueError(f"Expected {expected_count} curated facts, found {len(facts)}")
    if facts["amendment_fact_id"].duplicated().any():
        raise ValueError("Apple case contains duplicate amendment fact identifiers")
    if facts["metric_label"].isna().any():
        raise ValueError("Apple case contains an unmapped signal metric")
    if not facts["comparison_status"].eq("changed").all():
        raise ValueError("Curated Apple case must contain changed facts only")
    if facts[["original_accepted_at", "amendment_accepted_at", "period_end"]].isna().any().any():
        raise ValueError("Apple case contains an invalid timestamp")
    if not (facts["original_accepted_at"] < facts["amendment_accepted_at"]).all():
        raise ValueError("Apple case violates point-in-time filing order")
    if not facts["original_accession"].eq(str(metadata["original_accession"])).all():
        raise ValueError("Apple case original accession does not match metadata")
    if not facts["amendment_accession"].eq(str(metadata["amendment_accession"])).all():
        raise ValueError("Apple case amendment accession does not match metadata")


def _derive_gross_margin(facts: pd.DataFrame) -> pd.DataFrame:
    keys = ["period_start", "period_end", "fiscal_year"]
    gross_profit = facts.loc[
        facts["tag"].eq("GrossProfit"),
        [*keys, "original_value", "amendment_value"],
    ]
    revenue = facts.loc[
        facts["tag"].eq("SalesRevenueNet"),
        [*keys, "original_value", "amendment_value"],
    ]
    merged = gross_profit.merge(revenue, on=keys, suffixes=("_gross_profit", "_revenue"))
    if (
        len(merged) != 3
        or (
            merged[
                [
                    "original_value_revenue",
                    "amendment_value_revenue",
                ]
            ]
            == 0
        )
        .any()
        .any()
    ):
        raise ValueError("Apple case cannot derive the expected three-year gross-margin series")
    merged["original_margin"] = (
        merged["original_value_gross_profit"] / merged["original_value_revenue"]
    )
    merged["amendment_margin"] = (
        merged["amendment_value_gross_profit"] / merged["amendment_value_revenue"]
    )
    merged = merged.sort_values("fiscal_year").reset_index(drop=True)
    merged["original_yoy_change"] = merged["original_margin"].diff()
    merged["amendment_yoy_change"] = merged["amendment_margin"].diff()
    merged["signal_inflation"] = merged["amendment_yoy_change"] - merged["original_yoy_change"]
    return merged
