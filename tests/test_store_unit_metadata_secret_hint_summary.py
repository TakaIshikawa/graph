from __future__ import annotations

from graph.store.unit_metadata_secret_hint_summary import summarize_unit_metadata_secret_hints


def test_summarize_unit_metadata_secret_hints_redacts_nested_values():
    summary = summarize_unit_metadata_secret_hints(
        [
            {"id": "u1", "metadata": {"auth": {"api_key": "sk-test1234567890abcdef"}, "note": "token bucket"}},
            {"id": "u2", "metadata": {"checksum": "abcdef1234567890abcdef1234567890"}},
        ]
    )

    assert summary["affected_units"] == 2
    assert {"hint_type": "suspicious_key_path", "count": 1} in summary["hint_type_counts"]
    assert {"hint_type": "sk_prefix", "count": 1} in summary["hint_type_counts"]
    assert summary["key_path_counts"][0] == {"key_path": "auth.api_key", "count": 2}
    assert all("sk-test1234567890abcdef" not in example["redacted_value"] for example in summary["examples"])
