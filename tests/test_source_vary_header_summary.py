from __future__ import annotations

from graph.store import summarize_source_vary_headers


class SourceObject:
    def __init__(self, source_id: str, metadata: dict[str, object]) -> None:
        self.source_id = source_id
        self.metadata = metadata


def test_vary_summary_splits_normalizes_and_deduplicates_tokens():
    summary = summarize_source_vary_headers(
        [
            {"id": "a", "headers": {"Vary": "Accept-Encoding, Accept-Language, accept-encoding"}},
            {"id": "b", "vary": "Origin"},
            {"id": "c"},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_vary"] == 2
    assert summary["missing_vary_count"] == 1
    assert summary["wildcard_vary_count"] == 0
    assert summary["token_counts"] == {"accept-encoding": 1, "accept-language": 1, "origin": 1}
    assert summary["rows"] == [
        {
            "token": "accept-encoding",
            "count": 1,
            "source_ids": ["a"],
            "examples": ["Accept-Encoding, Accept-Language, accept-encoding"],
        },
        {
            "token": "accept-language",
            "count": 1,
            "source_ids": ["a"],
            "examples": ["Accept-Encoding, Accept-Language, accept-encoding"],
        },
        {"token": "origin", "count": 1, "source_ids": ["b"], "examples": ["Origin"]},
    ]


def test_vary_summary_reports_wildcard_rows():
    summary = summarize_source_vary_headers(
        [
            {"source_id": "wild", "Vary": "*"},
            {"source_id": "mixed", "response_headers": {"vary": "Accept, *"}},
        ]
    )

    assert summary["sources_with_vary"] == 2
    assert summary["wildcard_vary_count"] == 2
    assert summary["token_counts"] == {"*": 2, "accept": 1}
    assert {"token": "*", "count": 2, "source_ids": ["wild", "mixed"], "examples": ["*", "Accept, *"]} in summary["rows"]


def test_vary_summary_reads_metadata_and_object_sources():
    summary = summarize_source_vary_headers(
        [
            {"id": "metadata-field", "metadata": {"vary": "Cookie"}},
            {"id": "metadata-headers", "metadata": {"response_headers": {"VARY": "Accept_Encoding"}}},
            SourceObject("object", {"headers": {"Vary": "User-Agent"}}),
        ]
    )

    assert summary["sources_with_vary"] == 3
    assert summary["token_counts"] == {"accept-encoding": 1, "cookie": 1, "user-agent": 1}
    assert {row["token"]: row["source_ids"] for row in summary["rows"]} == {
        "accept-encoding": ["metadata-headers"],
        "cookie": ["metadata-field"],
        "user-agent": ["object"],
    }


def test_vary_summary_respects_sample_limit_while_counting():
    summary = summarize_source_vary_headers(
        [
            {"id": "a", "Vary": "Origin"},
            {"id": "b", "Vary": "Origin"},
            {"id": "c", "Vary": "Origin"},
        ],
        sample_limit=2,
    )

    assert summary["token_counts"] == {"origin": 3}
    assert summary["rows"] == [{"token": "origin", "count": 3, "source_ids": ["a", "b"], "examples": ["Origin"]}]
