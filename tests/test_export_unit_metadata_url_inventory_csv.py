from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_metadata_url_inventory_csv import export_units_to_metadata_url_inventory_csv


def test_export_units_to_metadata_url_inventory_csv_traverses_nested_metadata_paths():
    rows = list(
        csv.DictReader(
            StringIO(
                export_units_to_metadata_url_inventory_csv(
                    [
                        {
                            "id": "u1",
                            "source": "takeout",
                            "metadata": {
                                "links": [{"home": "https://Example.com/a."}, {"docs": ["see http://docs.test/x"]}],
                                "count": 3,
                                "nested": {"plain": "not a url"},
                            },
                        }
                    ]
                )
            )
        )
    )

    assert rows == [
        {
            "unit_id": "u1",
            "source": "takeout",
            "metadata_key_path": "metadata.links[0].home",
            "url": "https://Example.com/a",
            "hostname": "example.com",
            "scheme": "https",
        },
        {
            "unit_id": "u1",
            "source": "takeout",
            "metadata_key_path": "metadata.links[1].docs[0]",
            "url": "http://docs.test/x",
            "hostname": "docs.test",
            "scheme": "http",
        },
    ]


def test_export_units_to_metadata_url_inventory_csv_handles_www_urls():
    rows = list(csv.DictReader(StringIO(export_units_to_metadata_url_inventory_csv([{"id": "u1", "metadata": {"url": "www.Example.org/x"}}]))))

    assert rows[0]["hostname"] == "www.example.org"
    assert rows[0]["scheme"] == "https"
