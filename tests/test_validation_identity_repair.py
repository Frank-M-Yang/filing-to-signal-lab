"""Tests for outcome-blind validation identity repairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sec_data.validation_identity_repair import (
    calculate_validation_fallback_candidates,
    calculate_validation_identity_repair_components,
)


def test_missing_figi_uses_non_nan_composite_identity() -> None:
    targets = _repair_targets()
    cover = _cover_security()
    original = pd.DataFrame(
        [
            {
                "target_id": "anchor-1",
                "query_active": True,
                "security_type": "CS",
                "ticker": "TT",
                "exchange_mic": "XNYS",
                "provider_security_id": np.nan,
                "provider_security_id_kind": np.nan,
            }
        ]
    )
    fallback = pd.DataFrame(
        columns=[
            "identity_anchor_id",
            "ticker",
            "exchange_mic",
            "provider_security_id",
            "provider_security_id_kind",
            "provider_cik_status",
            "evidence_source",
        ]
    )

    result = calculate_validation_identity_repair_components(targets, cover, original, fallback)

    assert result.loc[0, "component_identity_ready"]
    assert result.loc[0, "identity_key_kind"] == "sec_cik_ticker_exchange_composite"
    assert result.loc[0, "identity_key"] != "nan"
    assert pd.isna(result.loc[0, "provider_security_id"])


def test_missing_provider_cik_requires_exact_sec_name_ticker_and_figi(
    tmp_path: Path,
) -> None:
    payload = {
        "status": "OK",
        "results": [
            {
                "active": True,
                "cik": None,
                "ticker": "GLXZ",
                "name": "GALAXY GAMING INC",
                "type": "CS",
                "primary_exchange": "OTC Link",
                "share_class_figi": "BBG001T46GK3",
            }
        ],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    targets = pd.DataFrame(
        [
            {
                "target_id": "fallback-1",
                "identity_anchor_id": "anchor-1",
                "validation_event_id": "event-1",
                "cik": "0000013156",
                "query_ticker": "GLXZ",
                "query_date": "2024-11-09",
                "anchor_role": "first_candidate",
            }
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "target_id": "fallback-1",
                "query_active": True,
                "path": str(path),
                "sha256": "a" * 64,
            }
        ]
    )

    candidates, rejections = calculate_validation_fallback_candidates(
        targets,
        snapshots,
        {"0000013156": "Galaxy Gaming, Inc."},
    )

    assert len(candidates) == 1
    assert candidates.loc[0, "provider_cik_status"] == "missing_name_exact"
    assert candidates.loc[0, "provider_security_id"] == "BBG001T46GK3"
    assert candidates.loc[0, "exchange_mic"] == "OTCM"
    assert rejections.empty


def _repair_targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "identity_anchor_id": "anchor-1",
                "validation_event_id": "event-1",
                "cik": "0001466258",
                "anchor_role": "first_candidate",
                "query_date": "2024-10-30",
            }
        ]
    )


def _cover_security() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "identity_anchor_id": "anchor-1",
                "common_equity_candidate": True,
                "security_subject_cik_match": True,
                "trading_symbol": "TT",
                "normalized_exchange_mic": "XNYS",
                "security_title": "Ordinary Shares, Par Value $1.00 per Share",
            }
        ]
    )
