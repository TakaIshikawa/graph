from __future__ import annotations

import csv
from datetime import date
from io import StringIO

from graph.export.metadata_value_type_csv import export_metadata_value_type_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, metadata: object) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_metadata_value_type_profiles_values_by_key():
    text = export_metadata_value_type_csv(
        [
            unit("a", {"status": "open", "score": 1, "flag": True, "tags": ["a"], "props": {"a": 1}}),
            unit("b", {"status": "open", "score": 2.5, "flag": False, "tags": (), "props": {}}),
            unit("c", {"status": None, "score": True, "extra": date(2026, 5, 13)}),
        ]
    )

    assert text.splitlines()[0] == (
        "metadata_key,unit_count,null_count,string_count,number_count,boolean_count,"
        "list_count,object_count,other_count,distinct_scalar_values"
    )
    assert rows(text) == [
        {
            "metadata_key": "extra",
            "unit_count": "1",
            "null_count": "0",
            "string_count": "0",
            "number_count": "0",
            "boolean_count": "0",
            "list_count": "0",
            "object_count": "0",
            "other_count": "1",
            "distinct_scalar_values": "0",
        },
        {
            "metadata_key": "flag",
            "unit_count": "2",
            "null_count": "0",
            "string_count": "0",
            "number_count": "0",
            "boolean_count": "2",
            "list_count": "0",
            "object_count": "0",
            "other_count": "0",
            "distinct_scalar_values": "2",
        },
        {
            "metadata_key": "props",
            "unit_count": "2",
            "null_count": "0",
            "string_count": "0",
            "number_count": "0",
            "boolean_count": "0",
            "list_count": "0",
            "object_count": "2",
            "other_count": "0",
            "distinct_scalar_values": "0",
        },
        {
            "metadata_key": "score",
            "unit_count": "3",
            "null_count": "0",
            "string_count": "0",
            "number_count": "2",
            "boolean_count": "1",
            "list_count": "0",
            "object_count": "0",
            "other_count": "0",
            "distinct_scalar_values": "3",
        },
        {
            "metadata_key": "status",
            "unit_count": "3",
            "null_count": "1",
            "string_count": "2",
            "number_count": "0",
            "boolean_count": "0",
            "list_count": "0",
            "object_count": "0",
            "other_count": "0",
            "distinct_scalar_values": "2",
        },
        {
            "metadata_key": "tags",
            "unit_count": "2",
            "null_count": "0",
            "string_count": "0",
            "number_count": "0",
            "boolean_count": "0",
            "list_count": "2",
            "object_count": "0",
            "other_count": "0",
            "distinct_scalar_values": "0",
        },
    ]


def test_metadata_value_type_ignores_blank_keys_and_non_mapping_metadata():
    text = export_metadata_value_type_csv(
        [
            unit("a", {"": 1, " name\nkey ": "Alpha"}),
            unit("b", None),
            unit("c", ["not", "mapping"]),
        ]
    )

    assert rows(text) == [
        {
            "metadata_key": "name key",
            "unit_count": "1",
            "null_count": "0",
            "string_count": "1",
            "number_count": "0",
            "boolean_count": "0",
            "list_count": "0",
            "object_count": "0",
            "other_count": "0",
            "distinct_scalar_values": "1",
        }
    ]


def test_metadata_value_type_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "metadata-types.csv"
    units = [unit("a", {"status": "open", "count": 1})]

    expected = export_metadata_value_type_csv(units)
    stats = export_metadata_value_type_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "metadata_key_count": 2,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }


def test_metadata_value_type_is_deterministic_for_reversed_input():
    units = [
        unit("a", {"z": "last", "a": 1}),
        unit("b", {"a": 2, "m": False}),
    ]

    assert export_metadata_value_type_csv(units) == export_metadata_value_type_csv(reversed(units))
