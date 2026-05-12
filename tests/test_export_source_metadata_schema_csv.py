from __future__ import annotations

import csv
from io import StringIO

from graph.export.source_metadata_schema_csv import export_source_metadata_schema_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_source_metadata_schema_csv_empty_input_has_header_only():
    assert export_source_metadata_schema_csv([]) == (
        "source_project,source_entity_type,metadata_key,unit_count,populated_unit_count,"
        "coverage_percent,observed_type_names,example_values\n"
    )


def test_source_metadata_schema_csv_groups_keys_by_source_and_entity_type():
    text = export_source_metadata_schema_csv(
        [
            unit("a", source_project="Project A", source_entity_type="task", metadata={"status": "open"}),
            unit("b", source_project="Project A", source_entity_type="task", metadata={"status": "done"}),
            unit("c", source_project="Project A", source_entity_type="note", metadata={"status": "draft"}),
            unit("d", source_project="Project B", source_entity_type="task", metadata={"status": "blocked"}),
        ]
    )

    assert rows(text) == [
        {
            "source_project": "Project A",
            "source_entity_type": "note",
            "metadata_key": "status",
            "unit_count": "1",
            "populated_unit_count": "1",
            "coverage_percent": "100.00",
            "observed_type_names": "str",
            "example_values": "draft",
        },
        {
            "source_project": "Project A",
            "source_entity_type": "task",
            "metadata_key": "status",
            "unit_count": "2",
            "populated_unit_count": "2",
            "coverage_percent": "100.00",
            "observed_type_names": "str",
            "example_values": "done; open",
        },
        {
            "source_project": "Project B",
            "source_entity_type": "task",
            "metadata_key": "status",
            "unit_count": "1",
            "populated_unit_count": "1",
            "coverage_percent": "100.00",
            "observed_type_names": "str",
            "example_values": "blocked",
        },
    ]


def test_source_metadata_schema_csv_coverage_uses_populated_values_in_group():
    text = export_source_metadata_schema_csv(
        [
            unit("a", metadata={"priority": "high", "empty": ""}),
            unit("b", metadata={"priority": "", "empty": []}),
            unit("c", metadata={"priority": None}),
            unit("d", metadata={}),
        ]
    )

    by_key = {row["metadata_key"]: row for row in rows(text)}
    assert by_key["priority"] == {
        "source_project": "max",
        "source_entity_type": "note",
        "metadata_key": "priority",
        "unit_count": "4",
        "populated_unit_count": "1",
        "coverage_percent": "25.00",
        "observed_type_names": "null; str",
        "example_values": "high",
    }
    assert by_key["empty"]["coverage_percent"] == "0.00"
    assert by_key["empty"]["observed_type_names"] == "list; str"


def test_source_metadata_schema_csv_limits_examples_deterministically():
    text = export_source_metadata_schema_csv(
        [
            unit("a", metadata={"label": "zeta"}),
            unit("b", metadata={"label": "alpha"}),
            unit("c", metadata={"label": "beta"}),
            unit("d", metadata={"label": "gamma"}),
        ]
    )

    assert rows(text)[0]["example_values"] == "alpha; beta; gamma"
    assert export_source_metadata_schema_csv(
        [
            unit("d", metadata={"label": "gamma"}),
            unit("c", metadata={"label": "beta"}),
            unit("b", metadata={"label": "alpha"}),
            unit("a", metadata={"label": "zeta"}),
        ]
    ) == text


def test_source_metadata_schema_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "source-metadata-schema.csv"
    units = [unit("a", metadata={"status": "open"})]

    expected = export_source_metadata_schema_csv(units)
    stats = export_source_metadata_schema_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "schema_key_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
