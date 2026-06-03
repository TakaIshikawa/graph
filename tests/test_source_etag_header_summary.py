from __future__ import annotations

from graph.store import summarize_source_etag_headers


class SourceObject:
    def __init__(self, source_id: str, metadata: dict[str, object]) -> None:
        self.source_id = source_id
        self.metadata = metadata


def test_etag_header_summary_counts_strong_weak_missing_and_distinct_tags():
    summary = summarize_source_etag_headers(
        [
            {"id": "strong", "etag": '"abc"'},
            {"id": "weak", "ETag": 'W/"abc"'},
            {"id": "missing"},
        ]
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_etag"] == 2
    assert summary["missing_etag_count"] == 1
    assert summary["weak_etag_count"] == 1
    assert summary["strong_etag_count"] == 1
    assert summary["distinct_etag_count"] == 2
    assert summary["rows"] == [
        {"etag": '"abc"', "validator_type": "strong", "count": 1, "source_ids": ["strong"], "examples": ['"abc"']},
        {"etag": 'W/"abc"', "validator_type": "weak", "count": 1, "source_ids": ["weak"], "examples": ['W/"abc"']},
    ]


def test_etag_header_summary_reads_metadata_and_header_containers_case_insensitively():
    summary = summarize_source_etag_headers(
        [
            {"id": "metadata-field", "metadata": {"e_tag": '"meta"'}},
            {"id": "source-headers", "headers": {"ETAG": '"headers"'}},
            {"id": "source-response", "response_headers": {"e_tag": 'W/"response"'}},
            SourceObject("object-metadata-headers", {"headers": {"E-Tag": '"object"'}}),
            {"id": "metadata-response", "metadata": {"response_headers": {"etag": '"nested"'}}},
        ]
    )

    assert summary["sources_with_etag"] == 5
    assert summary["weak_etag_count"] == 1
    assert {row["etag"]: row["source_ids"] for row in summary["rows"]} == {
        '"headers"': ["source-headers"],
        '"meta"': ["metadata-field"],
        '"nested"': ["metadata-response"],
        '"object"': ["object-metadata-headers"],
        'W/"response"': ["source-response"],
    }


def test_etag_header_summary_groups_duplicate_values_and_respects_sample_limit():
    summary = summarize_source_etag_headers(
        [
            {"id": "a", "headers": {"ETag": '"same"'}},
            {"id": "b", "metadata": {"ETag": '"same"'}},
            {"id": "c", "response_headers": {"ETag": '"same"'}},
            {"id": "d", "ETag": 'W/"same"'},
        ],
        sample_limit=2,
    )

    assert summary["sources_with_etag"] == 4
    assert summary["distinct_etag_count"] == 2
    assert summary["strong_etag_count"] == 3
    assert summary["weak_etag_count"] == 1
    assert summary["rows"] == [
        {"etag": '"same"', "validator_type": "strong", "count": 3, "source_ids": ["a", "b"], "examples": ['"same"']},
        {"etag": 'W/"same"', "validator_type": "weak", "count": 1, "source_ids": ["d"], "examples": ['W/"same"']},
    ]


def test_etag_header_summary_allows_zero_sample_limit_while_counting():
    summary = summarize_source_etag_headers(
        [{"id": "a", "ETag": '"same"'}, {"id": "b", "ETag": '"same"'}],
        sample_limit=0,
    )

    assert summary["rows"] == [{"etag": '"same"', "validator_type": "strong", "count": 2, "source_ids": [], "examples": []}]
