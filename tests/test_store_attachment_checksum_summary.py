from __future__ import annotations

from graph.store.attachment_checksum_summary import summarize_attachment_checksums


def test_summarize_attachment_checksums_counts_duplicates_algorithms_and_missing():
    summary = summarize_attachment_checksums(
        [
            {
                "id": "u1",
                "metadata": {
                    "attachments": [
                        {"id": "a1", "checksum": "ABC", "checksum_algorithm": "SHA-256"},
                        {"id": "a2"},
                    ]
                },
            },
            {
                "id": "u2",
                "metadata": {
                    "attachments": [
                        {"id": "b1", "sha256": "abc"},
                        {"id": "b2", "md5": "def"},
                    ]
                },
            },
        ]
    )

    assert summary == {
        "total_attachments": 4,
        "checksummed_attachments": 3,
        "missing_checksum_count": 1,
        "duplicate_checksum_groups": [
            {
                "checksum": "abc",
                "attachments": [
                    {"checksum": "abc", "unit_id": "u1", "attachment_id": "a1"},
                    {"checksum": "abc", "unit_id": "u2", "attachment_id": "b1"},
                ],
            }
        ],
        "algorithms_used": [{"algorithm": "md5", "count": 1}, {"algorithm": "sha256", "count": 2}],
        "largest_duplicate_group_size": 2,
    }


def test_summarize_attachment_checksums_handles_units_without_attachments():
    summary = summarize_attachment_checksums([{"id": "empty", "metadata": {}}])

    assert summary["total_attachments"] == 0
    assert summary["duplicate_checksum_groups"] == []
