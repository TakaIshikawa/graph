from __future__ import annotations

from graph.rag.result_version_signals import extract_result_version_signals


def test_extracts_version_signals_and_deduplicates_within_result():
    report = extract_result_version_signals(
        [
            {
                "id": "r1",
                "title": "API v2 and release 2025-04-01",
                "content": "Use package 1.2.3 with GPT-4.1 preview. API v2 remains supported.",
                "metadata": {"model": "gpt-4.1"},
            }
        ]
    )

    signals = report["results"][0]["signals"]
    assert report["results_with_signals"] == 1
    assert report["signal_type_counts"]["semantic_version"] == 1
    assert report["signal_type_counts"]["api_version"] == 1
    assert report["signal_type_counts"]["date_version"] == 1
    assert [signal["value"].casefold() for signal in signals].count("api v2") == 0
    assert len({(signal["type"], signal["value"].casefold()) for signal in signals}) == len(signals)


def test_text_fields_can_be_overridden():
    report = extract_result_version_signals([{"id": "r1", "body": "Release candidate rc1 for v3.2.1"}], text_fields=["body"])

    assert report["signal_type_counts"]["semantic_version"] == 1
    assert report["signal_type_counts"]["release_label"] == 1
