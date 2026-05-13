from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_metadata_presence_csv import export_unit_metadata_presence_csv
from graph.types.enums import ContentType
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: object = None, source_project: str = "A") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=unit_id,
        source_entity_type="note",
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={} if metadata is None else metadata,
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_metadata_presence_csv_empty_input_has_header_only():
    assert export_unit_metadata_presence_csv([]) == (
        "unit_id,source_project,source_entity_type,metadata_key_count,present_keys,missing_keys,completeness_ratio\n"
    )


def test_unit_metadata_presence_csv_derives_stable_union_of_metadata_keys():
    text = export_unit_metadata_presence_csv(
        [
            unit("b", metadata={"beta": 1, "alpha": 2}, source_project="B"),
            unit("a", metadata={"alpha": 3}),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "source_project": "A",
            "source_entity_type": "note",
            "metadata_key_count": "1",
            "present_keys": "alpha",
            "missing_keys": "beta",
            "completeness_ratio": "0.50",
        },
        {
            "unit_id": "b",
            "source_project": "B",
            "source_entity_type": "note",
            "metadata_key_count": "2",
            "present_keys": "alpha; beta",
            "missing_keys": "",
            "completeness_ratio": "1.00",
        },
    ]


def test_unit_metadata_presence_csv_explicit_keys_preserve_caller_intent_for_ratio():
    text = export_unit_metadata_presence_csv([unit("a", metadata={"beta": 1})], keys=["beta", "alpha", "beta"])

    assert rows(text)[0]["present_keys"] == "beta"
    assert rows(text)[0]["missing_keys"] == "alpha"
    assert rows(text)[0]["completeness_ratio"] == "0.50"


def test_unit_metadata_presence_csv_treats_non_dict_metadata_as_empty():
    text = export_unit_metadata_presence_csv([unit("a", metadata=["bad"])], keys=["alpha"])

    assert rows(text)[0]["metadata_key_count"] == "0"
    assert rows(text)[0]["missing_keys"] == "alpha"


def test_unit_metadata_presence_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "unit-metadata-presence.csv"
    units = [unit("a", metadata={"alpha": 1})]

    expected = export_unit_metadata_presence_csv(units, keys=["alpha"])
    stats = export_unit_metadata_presence_csv(units, path, keys=["alpha"])

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "metadata_key_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
