from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.unit_required_metadata_audit_csv import export_unit_required_metadata_audit_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str | None = SourceProject.MAX,
    source_entity_type: str | None = "note",
    title: str | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=title or f"Title {unit_id}",
        content="content",
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_required_metadata_audit_csv_empty_input_has_header_only():
    assert export_unit_required_metadata_audit_csv([], ["status"]) == (
        "unit_id,title,source_project,source_entity_type,metadata_key,value_state,available_metadata_keys\n"
    )


def test_unit_required_metadata_audit_csv_omits_units_with_all_required_metadata_populated():
    text = export_unit_required_metadata_audit_csv(
        [
            unit("a", metadata={"status": "open", "priority": "high"}),
            unit("b", metadata={"status": "done", "priority": "low"}),
        ],
        ["status", "priority"],
    )

    assert rows(text) == []


def test_unit_required_metadata_audit_csv_reports_missing_and_empty_keys():
    text = export_unit_required_metadata_audit_csv(
        [
            unit("a", metadata={"status": "open", "priority": ""}),
            unit("b", metadata={"status": [], "owner": "team"}),
            unit("c", metadata={"priority": None}),
            unit("d", metadata={"status": "done", "priority": "high"}),
        ],
        ["status", "priority"],
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Title a",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "priority",
            "value_state": "empty",
            "available_metadata_keys": "priority; status",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "priority",
            "value_state": "missing",
            "available_metadata_keys": "owner; status",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "status",
            "value_state": "empty",
            "available_metadata_keys": "owner; status",
        },
        {
            "unit_id": "c",
            "title": "Title c",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "priority",
            "value_state": "empty",
            "available_metadata_keys": "priority",
        },
        {
            "unit_id": "c",
            "title": "Title c",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "status",
            "value_state": "missing",
            "available_metadata_keys": "priority",
        },
    ]


@pytest.mark.parametrize("empty_value", [None, "", "   ", [], (), set(), {}])
def test_unit_required_metadata_audit_csv_treats_empty_values_as_empty(empty_value):
    text = export_unit_required_metadata_audit_csv([unit("a", metadata={"status": empty_value})], ["status"])

    assert rows(text)[0]["value_state"] == "empty"


def test_unit_required_metadata_audit_csv_normalizes_required_keys_deterministically():
    text = export_unit_required_metadata_audit_csv(
        [
            unit("b", metadata={"Owner": "team"}),
            unit("a", metadata={"Status": "", "status": "open"}),
        ],
        [" status ", "Status", "Owner", "status", "Owner", ""],
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Title a",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "Owner",
            "value_state": "missing",
            "available_metadata_keys": "Status; status",
        },
        {
            "unit_id": "a",
            "title": "Title a",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "Status",
            "value_state": "empty",
            "available_metadata_keys": "Status; status",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "Status",
            "value_state": "missing",
            "available_metadata_keys": "Owner",
        },
        {
            "unit_id": "b",
            "title": "Title b",
            "source_project": "max",
            "source_entity_type": "note",
            "metadata_key": "status",
            "value_state": "missing",
            "available_metadata_keys": "Owner",
        },
    ]


def test_unit_required_metadata_audit_csv_rejects_empty_required_keys():
    with pytest.raises(ValueError, match="required_keys"):
        export_unit_required_metadata_audit_csv([], [])

    with pytest.raises(ValueError, match="required_keys"):
        export_unit_required_metadata_audit_csv([], ["", "   "])


def test_unit_required_metadata_audit_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-required-metadata-audit.csv"
    units = [unit("a", metadata={"status": ""})]

    expected = export_unit_required_metadata_audit_csv(units, ["status", "owner"])
    stats = export_unit_required_metadata_audit_csv(units, ["status", "owner"], path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "required_key_count": 2,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }
