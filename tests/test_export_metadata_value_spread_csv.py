from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_metadata_value_spread_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_metadata_value_spread_empty_input_returns_header_only_csv():
    assert export_metadata_value_spread_csv([]) == (
        "metadata_key,unit_count,value_count,distinct_value_count,top_value,top_value_count,top_value_percent\n"
    )


def test_metadata_value_spread_summarizes_values_by_key():
    text = export_metadata_value_spread_csv(
        [
            unit("a", {"status": "done", "priority": ["high", "low"]}),
            unit("b", {"status": "done", "priority": ("high", "high")}),
            unit("c", {"status": "todo", "priority": {"low", "medium"}}),
        ]
    )

    assert rows(text) == [
        {
            "metadata_key": "priority",
            "unit_count": "3",
            "value_count": "5",
            "distinct_value_count": "3",
            "top_value": "high",
            "top_value_count": "2",
            "top_value_percent": "40.00",
        },
        {
            "metadata_key": "status",
            "unit_count": "3",
            "value_count": "3",
            "distinct_value_count": "2",
            "top_value": "done",
            "top_value_count": "2",
            "top_value_percent": "66.67",
        },
    ]


def test_metadata_value_spread_skips_empty_keys_and_values():
    text = export_metadata_value_spread_csv(
        [
            unit("a", {"": "ignored", "empty": "", "none": None}),
            unit("b", {" present ": [" value ", "", None]}),
        ]
    )

    assert rows(text) == [
        {
            "metadata_key": "present",
            "unit_count": "1",
            "value_count": "1",
            "distinct_value_count": "1",
            "top_value": "value",
            "top_value_count": "1",
            "top_value_percent": "100.00",
        }
    ]


def test_metadata_value_spread_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "metadata-spread.csv"
    units = [unit("a", {"status": "done"})]

    expected = export_metadata_value_spread_csv(units)
    stats = export_metadata_value_spread_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "metadata_key_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_metadata_value_spread_is_deterministic_for_reversed_input():
    units = [
        unit("a", {"z": "2"}),
        unit("b", {"a": "1"}),
        unit("c", {"a": "2"}),
    ]

    assert export_metadata_value_spread_csv(units) == export_metadata_value_spread_csv(reversed(units))
