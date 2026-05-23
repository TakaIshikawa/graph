from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_contact_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_contact_inventory_empty_input_has_header():
    assert export_unit_contact_inventory_csv([]) == (
        "unit_id,title,emails,phones,people,organizations,handles,contact_score\n"
    )


def test_unit_contact_inventory_extracts_metadata_and_content_contacts():
    text = export_unit_contact_inventory_csv(
        [
            {
                "id": "u1",
                "title": "One",
                "content": "Email a@example.com or call +1 555 123 4567. Ping @handle",
                "metadata": {
                    "emails": ["a@example.com", "b@example.com"],
                    "person": "Ada",
                    "organization": "Org",
                },
            }
        ]
    )

    result = rows(text)[0]
    assert result["emails"] == "a@example.com;b@example.com"
    assert result["people"] == "Ada"
    assert result["organizations"] == "Org"
    assert result["handles"] == "@handle"
    assert result["contact_score"] == "5"


def test_unit_contact_inventory_path_mode(tmp_path):
    path = tmp_path / "contacts.csv"
    stats = export_unit_contact_inventory_csv([{"unit_id": "u1", "metadata": {}}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["contact_score"] == "0"
    assert stats["unit_count"] == 1
