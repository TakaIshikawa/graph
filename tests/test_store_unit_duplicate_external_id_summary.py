from __future__ import annotations

from graph.store import summarize_unit_duplicate_external_ids


def test_duplicate_external_ids_are_scoped_by_source_and_key():
    report = summarize_unit_duplicate_external_ids(
        [
            {"id": "b", "source_project": "web", "source_id": "shared", "metadata": {"external_id": "x"}},
            {"id": "a", "source_project": "web", "source_id": "shared", "metadata": {"external_id": "x"}},
            {"id": "c", "source_project": "mail", "source_id": "shared", "metadata": {"external_id": "x"}},
            {"id": "d", "source_project": "web", "source_id": "unique", "metadata": {"external_id": "shared"}},
        ],
        sample_limit=1,
    )

    assert report["duplicate_group_count"] == 2
    assert report["duplicate_values"] == [
        {
            "source": "web",
            "identifier_key": "external_id",
            "identifier_value": "x",
            "unit_count": 2,
            "sample_unit_ids": ["a"],
        },
        {
            "source": "web",
            "identifier_key": "source_id",
            "identifier_value": "shared",
            "unit_count": 2,
            "sample_unit_ids": ["a"],
        },
    ]


def test_blank_identifiers_are_excluded_from_duplicates_but_counted_missing():
    report = summarize_unit_duplicate_external_ids(
        [
            {"id": "a", "source_project": "web", "source_id": ""},
            {"id": "b", "source_project": "web"},
        ]
    )

    assert report["duplicate_values"] == []
    missing = {
        (row["source"], row["identifier_key"]): row["missing_count"]
        for row in report["missing_identifier_counts"]
    }
    assert missing[("web", "source_id")] == 2
    assert missing[("web", "external_id")] == 2
