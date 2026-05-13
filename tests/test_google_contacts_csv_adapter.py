from __future__ import annotations

import csv

from graph.adapters.google_contacts_csv import GoogleContactsCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject


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

    unit = next(unit for unit in result.units if unit.source_entity_type == "contact")
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

    contacts = [unit for unit in result.units if unit.source_entity_type == "contact"]
    assert len(contacts) == 1
    assert contacts[0].title == "unnamed@example.com"
    assert contacts[0].source_id.startswith("google_contacts_csv:")


def test_google_contacts_csv_filters_and_registry(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(export, [{"Name": "Grace Hopper", "E-mail 1 - Value": "grace@example.com"}])

    assert GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["event"]).units == []
    assert get_adapter("google_contacts_csv", path=str(export)).name == "google_contacts_csv"


def test_google_contacts_csv_emits_organization_units_and_edges(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {
                "Name": "Ada Lovelace",
                "E-mail 1 - Value": "ada@engine.test",
                "Organization Name": "Analytical Engines",
                "Department": "Research",
                "Job Title": "Researcher",
            },
            {
                "Name": "Charles Babbage",
                "E-mail 1 - Value": "charles@engine.test",
                "Company": "Analytical Engines",
                "Department": "Engineering",
                "Job Title": "Founder",
            },
        ],
    )

    result = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["contact", "organization"])

    contacts = [unit for unit in result.units if unit.source_entity_type == "contact"]
    organizations = [unit for unit in result.units if unit.source_entity_type == "organization"]
    assert len(contacts) == 2
    assert len(organizations) == 1
    organization = organizations[0]
    assert organization.title == "Analytical Engines"
    assert organization.metadata["contact_count"] == 2
    assert organization.metadata["departments"] == ["Engineering", "Research"]
    assert organization.metadata["contact_source_ids"] == sorted(contact.source_id for contact in contacts)
    assert organization.metadata["email_domains"] == ["engine.test"]
    assert organization.metadata["source_files"] == ["contacts.csv"]
    assert len(result.edges) == 2
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert {edge.from_unit_id for edge in result.edges} == {organization.source_id}
    assert {edge.to_unit_id for edge in result.edges} == {contact.source_id for contact in contacts}

    org_only = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["organization"])
    assert [unit.source_entity_type for unit in org_only.units] == ["organization"]
    assert org_only.edges == []


def test_google_contacts_csv_emits_group_units_and_edges(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {
                "Name": "Ada Lovelace",
                "E-mail 1 - Value": "ada@example.com",
                "E-mail 2 - Value": "work@example.com",
                "Organization Name": "Analytical Engines",
                "Group Membership": "Friends, Research",
            },
            {
                "Name": "Grace Hopper",
                "E-mail 1 - Value": "grace@example.com",
                "Company": "Navy",
                "Groups": "Friends",
            },
        ],
    )

    result = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["contact", "group"])

    assert GoogleContactsCsvAdapter(path=str(export)).entity_types == ["contact", "organization", "group", "domain"]
    groups = sorted((unit for unit in result.units if unit.source_entity_type == "group"), key=lambda unit: unit.title)
    assert [unit.title for unit in groups] == ["Friends", "Research"]
    friends = groups[0]
    contacts = [unit for unit in result.units if unit.source_entity_type == "contact"]
    assert friends.metadata["contact_count"] == 2
    assert friends.metadata["contact_source_ids"] == sorted(unit.source_id for unit in contacts)
    assert friends.metadata["email_count"] == 3
    assert friends.metadata["organization_names"] == ["Analytical Engines", "Navy"]
    assert friends.metadata["source_files"] == ["contacts.csv"]
    assert len(result.edges) == 3

    group_only = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["group"])
    assert {unit.source_entity_type for unit in group_only.units} == {"group"}
    assert group_only.edges == []


def test_google_contacts_csv_parses_relationship_metadata_and_content(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {
                "Name": "Ada Lovelace",
                "Relation 1 - Type": "Spouse",
                "Relation 1 - Value": "Charles Babbage",
                "Relation 2 - Type": "Colleague",
                "Relation 2 - Value": "grace@example.com",
                "Relation 3 - Type": "spouse",
                "Relation 3 - Value": "Charles Babbage",
            }
        ],
    )

    result = GoogleContactsCsvAdapter(path=str(export)).ingest()

    contact = next(unit for unit in result.units if unit.source_entity_type == "contact")
    assert contact.metadata["relationships"] == [
        {"type": "Spouse", "value": "Charles Babbage"},
        {"type": "Colleague", "value": "grace@example.com"},
    ]
    assert "Relationships: Spouse: Charles Babbage, Colleague: grace@example.com" in contact.content


def test_google_contacts_csv_emits_relationship_edges_by_name_and_email(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {
                "Name": "Ada Lovelace",
                "E-mail 1 - Value": "ada@example.com",
                "Relation 1 - Type": "Spouse",
                "Relation 1 - Value": "charles babbage",
                "Relation 2 - Type": "Colleague",
                "Relation 2 - Value": "grace@example.com",
            },
            {"Name": "Charles Babbage", "E-mail 1 - Value": "charles@example.com"},
            {"Name": "Grace Hopper", "E-mail 1 - Value": "grace@example.com"},
        ],
    )

    first = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["contact"])
    second = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["contact"])

    contacts = {unit.title: unit for unit in first.units}
    edges = sorted(first.edges, key=lambda edge: edge.metadata["relationship_type"])
    assert len(edges) == 2
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]
    assert {edge.relation for edge in edges} == {EdgeRelation.REFERENCES}
    assert {edge.source for edge in edges} == {EdgeSource.SOURCE}
    assert {edge.from_unit_id for edge in edges} == {contacts["Ada Lovelace"].source_id}
    assert {edge.to_unit_id for edge in edges} == {
        contacts["Charles Babbage"].source_id,
        contacts["Grace Hopper"].source_id,
    }
    assert [edge.metadata for edge in edges] == [
        {
            "source_project": SourceProject.GOOGLE_CONTACTS_CSV.value,
            "from_entity_type": "contact",
            "to_entity_type": "contact",
            "relationship_type": "Colleague",
            "relationship_value": "grace@example.com",
        },
        {
            "source_project": SourceProject.GOOGLE_CONTACTS_CSV.value,
            "from_entity_type": "contact",
            "to_entity_type": "contact",
            "relationship_type": "Spouse",
            "relationship_value": "charles babbage",
        },
    ]


def test_google_contacts_csv_relationship_edges_respect_filters_and_skip_self_or_no_match(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {
                "Name": "Ada Lovelace",
                "E-mail 1 - Value": "ada@example.com",
                "Relation 1 - Type": "Self",
                "Relation 1 - Value": "ada@example.com",
                "Relation 2 - Type": "Friend",
                "Relation 2 - Value": "Unknown Person",
            },
            {"Name": "Grace Hopper", "E-mail 1 - Value": "grace@example.com"},
        ],
    )

    result = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["contact"])
    ada = next(unit for unit in result.units if unit.title == "Ada Lovelace")

    assert ada.metadata["relationships"] == [
        {"type": "Self", "value": "ada@example.com"},
        {"type": "Friend", "value": "Unknown Person"},
    ]
    assert result.edges == []
    assert GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["group"]).edges == []


def test_google_contacts_csv_emits_domain_units_and_edges(tmp_path):
    export = tmp_path / "contacts.csv"
    _write_csv(
        export,
        [
            {
                "Name": "Ada Lovelace",
                "E-mail 1 - Value": "ada@example.com",
                "E-mail 2 - Value": "ada@work.test",
                "Organization Name": "Analytical Engines",
            },
            {
                "Name": "Grace Hopper",
                "E-mail 1 - Value": "grace@example.com",
                "Company": "Navy",
            },
        ],
    )

    result = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["contact", "domain"])
    second = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["contact", "domain"])

    contacts = [unit for unit in result.units if unit.source_entity_type == "contact"]
    domains = sorted((unit for unit in result.units if unit.source_entity_type == "domain"), key=lambda unit: unit.title)
    assert [unit.title for unit in domains] == ["example.com", "work.test"]
    example = domains[0]
    assert example.metadata["contact_count"] == 2
    assert example.metadata["contact_source_ids"] == sorted(contact.source_id for contact in contacts)
    assert example.metadata["observed_organizations"] == ["Analytical Engines", "Navy"]
    assert example.metadata["source_files"] == ["contacts.csv"]
    assert [unit.source_id for unit in domains] == [
        unit.source_id for unit in sorted((u for u in second.units if u.source_entity_type == "domain"), key=lambda u: u.title)
    ]
    assert len(result.edges) == 3
    assert {edge.relation for edge in result.edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}
    assert {edge.from_unit_id for edge in result.edges} == {contact.source_id for contact in contacts}
    assert {edge.to_unit_id for edge in result.edges} == {domain.source_id for domain in domains}
    assert [edge.id for edge in result.edges] == [edge.id for edge in second.edges]

    domain_only = GoogleContactsCsvAdapter(path=str(export)).ingest(entity_types=["domain"])
    assert {unit.source_entity_type for unit in domain_only.units} == {"domain"}
    assert domain_only.edges == []
