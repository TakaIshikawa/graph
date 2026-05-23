from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_source_import_batch_timeline_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_import_batch_timeline_empty_input_has_header_only():
    assert export_source_import_batch_timeline_csv([]) == (
        "import_batch,imported_date,source_count,earliest_source_date,latest_source_date,source_names\n"
    )


def test_source_import_batch_timeline_groups_explicit_batches_by_imported_date():
    text = export_source_import_batch_timeline_csv(
        [
            {
                "id": "s1",
                "name": "Alpha",
                "import_batch": "batch-1",
                "imported_at": "2024-02-10T09:00:00Z",
                "metadata": {"published_at": "2024-01-01"},
            },
            {
                "id": "s2",
                "name": "Beta",
                "metadata": {
                    "import_batch": "batch-1",
                    "imported_at": "2024-02-10",
                    "source_date": "2024-01-05",
                },
            },
            {
                "id": "s3",
                "name": "Gamma",
                "import_batch": "batch-2",
                "imported_at": "2024-03-01",
                "date": "2024-02-20",
            },
        ]
    )

    assert rows(text) == [
        {
            "import_batch": "batch-1",
            "imported_date": "2024-02-10",
            "source_count": "2",
            "earliest_source_date": "2024-01-01",
            "latest_source_date": "2024-01-05",
            "source_names": "Alpha; Beta",
        },
        {
            "import_batch": "batch-2",
            "imported_date": "2024-03-01",
            "source_count": "1",
            "earliest_source_date": "2024-02-20",
            "latest_source_date": "2024-02-20",
            "source_names": "Gamma",
        },
    ]


def test_source_import_batch_timeline_uses_imported_at_only_grouping_and_unknown_labels():
    text = export_source_import_batch_timeline_csv(
        [
            {"source_id": "s1", "title": "One", "metadata": {"imported_at": "2024-04-01", "date": "bad"}},
            {"source_id": "s2", "title": "Two", "metadata": {"imported_at": "2024-04-01", "date": "2024-03-31"}},
            {"source_id": "s3", "title": "Three", "metadata": {"imported_at": "not-a-date", "published_date": "2024-01-01"}},
            {"source_id": "s4", "title": "Four", "metadata": {}},
        ]
    )

    assert rows(text) == [
        {
            "import_batch": "unknown_batch",
            "imported_date": "2024-04-01",
            "source_count": "2",
            "earliest_source_date": "2024-03-31",
            "latest_source_date": "2024-03-31",
            "source_names": "One; Two",
        },
        {
            "import_batch": "unknown_batch",
            "imported_date": "unknown_date",
            "source_count": "2",
            "earliest_source_date": "2024-01-01",
            "latest_source_date": "2024-01-01",
            "source_names": "Four; Three",
        },
    ]


def test_source_import_batch_timeline_path_mode(tmp_path):
    path = tmp_path / "reports" / "imports.csv"
    sources = [{"id": "s1", "name": "One", "metadata": {"import_batch": "batch", "imported_at": "2024-01-01"}}]

    expected = export_source_import_batch_timeline_csv(sources)
    stats = export_source_import_batch_timeline_csv(sources, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "source_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
