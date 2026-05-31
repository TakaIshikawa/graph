from __future__ import annotations

from dataclasses import dataclass

from graph.rag.context_personal_data_signal import analyze_context_personal_data_signals


@dataclass
class ContextObject:
    title: str
    text: str


def test_detects_common_personal_data_without_returning_values():
    report = analyze_context_personal_data_signals(
        [
            "Contact ada@example.com at 415-555-0100.",
            {"text": "Ship to 123 Main Street. API key: sk_live_1234567890abcdef."},
            ContextObject(title="Profile", text="SSN on file. Full name: Ada Lovelace."),
        ]
    )

    assert report["signal_counts"] == {
        "email": 1,
        "phone": 1,
        "physical_address": 1,
        "api_key_or_token": 1,
        "government_id_label": 1,
        "personal_name_label": 1,
    }
    assert report["risky_item_count"] == 3
    assert report["risk_level"] == "high"
    assert {"item_index": 1, "signal_type": "api_key_or_token"} in report["examples"]
    assert all("sk_live" not in str(example) for example in report["examples"])
    assert all("ada@example.com" not in str(example) for example in report["examples"])


def test_risk_level_increases_with_signal_accumulation():
    low = analyze_context_personal_data_signals(["Email: ada@example.com"])
    medium = analyze_context_personal_data_signals(["Email: ada@example.com", "Phone 415-555-0100"])
    high = analyze_context_personal_data_signals(
        ["Email: ada@example.com", "Phone 415-555-0100", "SSN required", "Customer name: Ada"]
    )

    assert low["risk_level"] == "low"
    assert medium["risk_level"] == "medium"
    assert high["risk_level"] == "high"


def test_handles_empty_context_without_signals():
    report = analyze_context_personal_data_signals([{"text": "Public product documentation."}])

    assert report["risky_item_count"] == 0
    assert report["risk_level"] == "low"
    assert report["examples"] == []
