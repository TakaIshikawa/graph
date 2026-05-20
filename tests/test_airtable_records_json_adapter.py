from __future__ import annotations

from datetime import datetime, timezone
import json

from graph.adapters.airtable_records_json import AirtableRecordsJsonAdapter


def test_airtable_records_json_imports_api_records_and_preserves_typed_metadata(tmp_path):
    export = tmp_path / "records.json"
    export.write_text(
        json.dumps(
            {
                "metadata": {"base": {"name": "Product Base"}, "tableName": "Roadmap"},
                "records": [
                    {
                        "id": "recAlpha123",
                        "createdTime": "2025-01-02T03:04:05.000Z",
                        "fields": {
                            "Name": "Launch plan",
                            "Priority": 2,
                            "Active": True,
                            "Tags": ["release", "customer"],
                            "Owner": {"name": "Ada", "team": {"name": "Ops"}},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = AirtableRecordsJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "airtable_records_json"
    assert unit.source_entity_type == "record"
    assert unit.source_id == "airtable_records_json:recAlpha123"
    assert unit.title == "Launch plan"
    assert unit.metadata["record_id"] == "recAlpha123"
    assert unit.metadata["createdTime"] == "2025-01-02T03:04:05.000Z"
    assert unit.metadata["created_time"] == "2025-01-02T03:04:05.000Z"
    assert unit.metadata["base_name"] == "Product Base"
    assert unit.metadata["table_name"] == "Roadmap"
    assert unit.metadata["source_file"] == "records.json"
    assert unit.metadata["source_row"] == 1
    assert unit.metadata["fields"]["Priority"] == 2
    assert unit.metadata["fields"]["Active"] is True
    assert unit.metadata["fields"]["Tags"] == ["release", "customer"]
    assert unit.metadata["fields"]["Owner.name"] == "Ada"
    assert unit.metadata["fields"]["Owner.team.name"] == "Ops"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_airtable_records_json_adapter_options_override_payload_context(tmp_path):
    export = tmp_path / "records.json"
    export.write_text(
        json.dumps(
            {
                "baseName": "Payload Base",
                "table": {"name": "Payload Table"},
                "records": [
                    {"id": "recTwo", "createdTime": "2025-01-03T00:00:00Z", "fields": {"Title": "Second"}}
                ],
            }
        ),
        encoding="utf-8",
    )

    unit = AirtableRecordsJsonAdapter(
        path=str(export),
        base_name="Option Base",
        table_name="Option Table",
    ).ingest().units[0]

    assert unit.metadata["base_name"] == "Option Base"
    assert unit.metadata["table_name"] == "Option Table"
    assert unit.metadata["title_field"] == "Title"


def test_airtable_records_json_filters_entity_types_and_since(tmp_path):
    export = tmp_path / "records.json"
    export.write_text(
        json.dumps(
            {
                "records": [
                    {"id": "recOld", "createdTime": "2025-01-01T00:00:00Z", "fields": {"Name": "Old"}},
                    {"id": "recNew", "createdTime": "2025-01-03T00:00:00Z", "fields": {"Name": "New"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    since = type("Sync", (), {"last_sync_at": datetime(2025, 1, 2, tzinfo=timezone.utc)})()

    adapter = AirtableRecordsJsonAdapter(path=str(export))

    assert [unit.title for unit in adapter.ingest(since=since).units] == ["New"]
    assert adapter.ingest(entity_types=["table"]).units == []
