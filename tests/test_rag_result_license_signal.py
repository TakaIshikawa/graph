from __future__ import annotations

from graph.rag.result_license_signal import analyze_result_license_signals


def test_result_license_signal_classifies_common_license_strings():
    summary = analyze_result_license_signals(
        [
            {"id": "cc", "title": "CC", "license": "Creative Commons BY 4.0"},
            {"id": "pd", "metadata": {"rights": "Public Domain"}},
            {"id": "arr", "copyright": "All rights reserved"},
            {"id": "missing", "title": "Missing"},
        ]
    )

    assert summary["license_counts"] == {"all_rights_reserved": 1, "creative_commons": 1, "public_domain": 1, "unknown": 1}
    assert summary["permissive_count"] == 2
    assert summary["restrictive_count"] == 1
    assert summary["unknown_count"] == 1
    assert summary["samples"][0] == {"result_id": "cc", "title": "CC", "license": "creative_commons"}
