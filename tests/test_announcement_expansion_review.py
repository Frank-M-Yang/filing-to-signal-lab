"""No weakened review or replacement of inconvenient frozen issuers is admitted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.review_announcement_expansion as runner


@pytest.mark.parametrize(
    "flag",
    [
        "returns_used_for_review",
        "first_public_certified",
        "trading_admission",
        "model_retuning_allowed",
        "rounding_plugs_allowed",
        "sample_replacement_allowed",
    ],
)
def test_review_rejects_weakened_contract_before_reading_evidence(
    flag: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(Path("docs/announcement_expansion_review_v1.json").read_bytes())
    config[flag] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid contract must fail before reading any evidence")

    monkeypatch.setattr(runner, "_json", forbidden)
    with pytest.raises(ValueError, match="weakened"):
        runner.review_batch(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("violation", ["missing", "duplicate", "replacement"])
def test_all_twenty_frozen_issuers_require_dispositions_without_replacement(
    violation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads(Path("docs/announcement_expansion_review_v1.json").read_bytes())
    selected = [
        {"assigned_ticker": r["ticker"]} for r in [*config["reviews"], *config["blocked_cases"]]
    ]
    if violation == "missing":
        config["blocked_cases"].pop()
    elif violation == "duplicate":
        config["blocked_cases"].append(config["blocked_cases"][0])
    else:
        config["blocked_cases"][0]["ticker"] = "REPLACEMENT"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    def metadata(root: Path, checksum: str, name: str) -> dict[str, Any]:
        if name == "plan.json":
            return {"selected_cases": selected}
        if name == "summary.json":
            return {"selection_plan_manifest_sha256": config["plan_manifest_sha256"]}
        raise AssertionError("Dispositions must fail before financial source inspection")

    monkeypatch.setattr(runner, "_json", metadata)
    with pytest.raises(ValueError, match="exactly one disposition"):
        runner.review_batch(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
