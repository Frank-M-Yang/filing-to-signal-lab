"""Tests for the aggregate-only research snapshot shipped in a public clone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def test_public_preview_is_aggregate_and_matches_frozen_verdict() -> None:
    """Keep the public fallback useful without redistributing event-level records."""
    path = Path(__file__).parents[1] / "data" / "examples" / "research_preview_summary.json"
    preview = json.loads(path.read_text(encoding="utf-8"))

    assert preview["schema_version"] == "public-research-preview-v1"
    effectiveness = preview["longitudinal_effectiveness"]
    assert effectiveness["complete_events"] == 49
    assert effectiveness["issuers"] == 15
    assert effectiveness["representation_ic"]["Innovation std"] == 0.044695
    assert effectiveness["survivor_only_inflation_pp"] == 1.8107
    assert effectiveness["verdict"] == "effectiveness_not_established"

    serialized_keys = set(_all_keys(preview))
    forbidden_event_fields = {
        "accession_number",
        "accepted_at",
        "api_key",
        "company_id",
        "provider_security_id",
        "ticker",
    }
    assert serialized_keys.isdisjoint(forbidden_event_fields)


def _all_keys(value: Any) -> list[str]:
    """Return every mapping key from a nested JSON-compatible object."""
    if isinstance(value, dict):
        keys = list(value)
        for child in value.values():
            keys.extend(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: list[str] = []
        for child in value:
            keys.extend(_all_keys(child))
        return keys
    return []
