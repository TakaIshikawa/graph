from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_metadata_url_scheme_csv import export_unit_metadata_url_scheme_csv


def test_export_unit_metadata_url_scheme_csv_traverses_nested_metadata_and_classifies_schemes():
    rows = list(
        csv.DictReader(
            StringIO(
                export_unit_metadata_url_scheme_csv(
                    [{"id": "u1", "title": "Meta", "metadata": {"links": [{"home": "https://a.test"}, {"file": "file:///tmp/a"}], "app": "obsidian://open", "relative": "/notes/a", "plain": "hello"}}]
                )
            )
        )
    )

    assert [(row["metadata_key"], row["scheme"], row["url"]) for row in rows] == [
        ("metadata.app", "obsidian", "obsidian://open"),
        ("metadata.links.0.home", "https", "https://a.test"),
        ("metadata.links.1.file", "file", "file:///tmp/a"),
        ("metadata.relative", "missing", "/notes/a"),
    ]
