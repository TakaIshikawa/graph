from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_unit_metadata_value_frequency_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, title: str | None = None, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title if title is not None else f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_metadata_value_frequency_csv_empty_input_returns_header():
    assert export_unit_metadata_value_frequency_csv([]) == (
        "source_project,source_entity_type,metadata_key,metadata_value,unit_count,unit_ids,sample_titles\n"
    )


def test_export_unit_metadata_value_frequency_csv_groups_scalar_values_deterministically():
    text = export_unit_metadata_value_frequency_csv(
        [
            unit("b", title="Beta", metadata={"status": "Done"}),
            unit("a", title="Alpha", metadata={"status": "Done"}),
            unit("c", title="Gamma", metadata={"status": "Todo"}),
        ]
    )

    result = rows(text)
    assert result[0]["metadata_value"] == "Done"
    assert result[0]["unit_count"] == "2"
    assert result[0]["unit_ids"] == "a;b"
    assert result[0]["sample_titles"] == "Alpha;Beta"
    assert result[1]["metadata_value"] == "Todo"


def test_export_unit_metadata_value_frequency_csv_handles_lists_dicts_and_blank_values():
    text = export_unit_metadata_value_frequency_csv(
        [
            unit("a", metadata={"tags": ["z", "x"], "payload": {"b": 2, "a": 1}, "empty": None}),
            unit("b", metadata={"tags": ["x"], "payload": {"a": 1, "b": 2}, "empty": ""}),
        ]
    )

    by_key_value = {(row["metadata_key"], row["metadata_value"]): row for row in rows(text)}
    assert by_key_value[("tags", "x")]["unit_count"] == "2"
    assert by_key_value[("payload", '{"a":1,"b":2}')]["unit_ids"] == "a;b"
    assert by_key_value[("empty", "")]["unit_count"] == "2"


def test_export_unit_metadata_value_frequency_csv_path_mode(tmp_path):
    path = tmp_path / "values.csv"
    stats = export_unit_metadata_value_frequency_csv([unit("a", metadata={"status": "new"})], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["metadata_key"] == "status"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
