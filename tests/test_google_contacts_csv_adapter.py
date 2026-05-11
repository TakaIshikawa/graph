from __future__ import annotations

import csv

from graph.adapters.google_contacts_csv import GoogleContactsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_google_contacts_csv_normalizes_repeated_fields(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {
                "Name": "Ada Lovelace",
                "Notes": "First programmer",
                "E-mail 1 - Value": "ada@example.com",
                "E-mail 2 - Value": "work@example.com",
                "Phone 1 - Value": "+1 555 0100",
                "Address 1 - Formatted": "1 Algorithm Ave",
                "Website 1 - Value": "https://ada.example",
                "Organization 1 - Name": "Analytical Engines",
                "Organization 1 - Title": "Researcher",
                "Birthday": "1815-12-10",
                "Group Membership": "* myContacts, Friends",
            }
        ],
    )

    result = GoogleContactsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOOGLE_CONTACTS_CSV
    assert unit.title == "Ada Lovelace"
    assert unit.metadata["emails"] == ["ada@example.com", "work@example.com"]
    assert unit.metadata["phones"] == ["+1 555 0100"]
    assert unit.metadata["addresses"] == ["1 Algorithm Ave"]
    assert unit.metadata["websites"] == ["https://ada.example"]
    assert unit.metadata["organization"]["name"] == "Analytical Engines"
    assert unit.metadata["organization"]["title"] == "Researcher"
    assert unit.metadata["groups"] == ["My Contacts", "Friends"]
    assert "Friends" in unit.tags


def test_google_contacts_csv_handles_empty_rows_and_missing_names(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {"Name": "", "E-mail 1 - Value": "", "Phone 1 - Value": "", "Group Membership": ""},
            {"Name": "", "E-mail 1 - Value": "unnamed@example.com", "Phone 1 - Value": "", "Group Membership": ""},
        ],
    )

    result = GoogleContactsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "unnamed@example.com"
    assert result.units[0].source_id.startswith("google_contacts_csv:")


def test_google_contacts_csv_filters_and_registry(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(export, [{"Name": "Grace Hopper", "E-mail 1 - Value": "grace@example.com"}])

    assert GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["event"]).units == []
    assert get_adapter("google_contacts_csv", path=str(export)).name == "google_contacts_csv"
