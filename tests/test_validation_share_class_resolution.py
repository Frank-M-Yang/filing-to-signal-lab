"""Tests for SEC-disclosed primary share-class resolution."""

from __future__ import annotations

import pandas as pd

from sec_data.validation_share_class_resolution import calculate_share_class_resolution


def test_largest_disclosed_class_is_selected_without_price_inputs() -> None:
    events = pd.DataFrame([_event("event-1"), _event("event-2")])
    components = pd.DataFrame(
        [
            _component(anchor, ticker, identity, title)
            for anchor in ("first_candidate", "last_candidate")
            for ticker, identity, title in (
                ("DUALA", "FIGI-A", "Common Stock Class A"),
                ("DUALB", "FIGI-B", "Common Stock Class B"),
            )
        ]
    )
    evidence = pd.DataFrame(
        [
            _security_evidence(target, anchor, ticker, title, digest)
            for target, anchor, digest in (
                ("endpoint-1", "first_candidate", "a" * 64),
                ("endpoint-2", "last_candidate", "b" * 64),
            )
            for ticker, title in (
                ("DUALA", "Common Stock Class A"),
                ("DUALB", "Common Stock Class B"),
            )
        ]
    )
    contents = {
        "endpoint-1": (_filing(5_200_000, 1_600_000, "2024-10-25"), "a" * 64),
        "endpoint-2": (_filing(5_300_000, 1_500_000, "2025-10-24"), "b" * 64),
    }

    resolved, facts, candidates, decisions, gates, report = calculate_share_class_resolution(
        events, components, evidence, contents
    )

    assert len(facts) == 4
    assert candidates["largest_disclosed_share_class"].sum() == 2
    assert decisions.loc[0, "selected_ticker"] == "DUALA"
    assert resolved["assigned_ticker"].eq("DUALA").all()
    assert resolved["security_component_count"].eq(1).all()
    assert (
        resolved["identity_assignment_method"]
        .eq("primary_share_class_by_largest_disclosed_shares")
        .all()
    )
    assert report["prior_price_pilot_used_as_selection_input"] is False
    assert report["multiclass_events_after_resolution"] == 0
    assert gates.loc[gates["gate"].ne("return_outcome_read"), "passed"].all()
    assert not gates.loc[gates["gate"].eq("return_outcome_read"), "passed"].any()


def _event(event_id: str) -> dict[str, object]:
    return {
        "market_event_id": event_id,
        "cik": "1",
        "event_security_identity_ready": True,
        "identity_assignment_method": "identity_repair_equal_weighted_share_class_basket",
        "assigned_ticker": "DUALA|DUALB",
        "assigned_exchange_mic": "XNAS",
        "assigned_provider_security_id": "FIGI-A|FIGI-B",
        "assigned_provider_security_id_kind": "",
        "security_component_count": 2,
        "assigned_security_identity_keys": "FIGI-A|FIGI-B",
        "issuer_return_construction": "equal_weighted_share_class_basket",
        "identity_repair_status": "stable_components_across_both_endpoints",
        "price_collection_authorized": False,
        "return_outcome_read_authorized": False,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }


def _component(anchor: str, ticker: str, identity: str, title: str) -> dict[str, object]:
    return {
        "cik": "1",
        "anchor_role": anchor,
        "ticker": ticker,
        "exchange_mic": "XNAS",
        "security_title": title,
        "provider_security_id": identity,
        "provider_security_id_kind": "share_class_figi",
        "identity_key": identity,
        "identity_key_kind": "share_class_figi",
        "component_identity_ready": True,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }


def _security_evidence(
    target: str, anchor: str, ticker: str, title: str, digest: str
) -> dict[str, object]:
    return {
        "target_id": target,
        "cik": "1",
        "anchor_role": anchor,
        "trading_symbol": ticker,
        "security_title": title,
        "primary_source_sha256": digest,
        "selection_uses_return_values": False,
        "outcome_values_loaded": False,
    }


def _filing(class_a: int, class_b: int, instant: str) -> bytes:
    return f"""<html><body>
    <xbrli:context id="a"><xbrli:entity><xbrli:identifier>1</xbrli:identifier>
    <xbrli:segment><xbrldi:explicitMember dimension="dei:StatementClassOfStockAxis">
    demo:CommonClassAMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
    <xbrli:period><xbrli:instant>{instant}</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:context id="b"><xbrli:entity><xbrli:identifier>1</xbrli:identifier>
    <xbrli:segment><xbrldi:explicitMember dimension="dei:StatementClassOfStockAxis">
    demo:CommonClassBMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
    <xbrli:period><xbrli:instant>{instant}</xbrli:instant></xbrli:period></xbrli:context>
    <ix:nonFraction name="dei:EntityCommonStockSharesOutstanding" contextRef="a">
    {class_a}</ix:nonFraction>
    <ix:nonFraction name="dei:EntityCommonStockSharesOutstanding" contextRef="b">
    {class_b}</ix:nonFraction>
    </body></html>""".encode()
