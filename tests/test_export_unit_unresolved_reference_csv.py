from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_unresolved_reference_csv import export_unit_unresolved_reference_csv
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str = "",
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project="Project A",
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        metadata=metadata or {},
        tags=[],
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_unresolved_reference_csv_empty_input_has_header_only():
    assert export_unit_unresolved_reference_csv([]) == (
        "unit_id,title,reference,normalized_reference,reference_count,matched_candidate_count\n"
    )


def test_unit_unresolved_reference_csv_reports_only_unmatched_references_deterministically():
    text = export_unit_unresolved_reference_csv(
        [
            unit("z", "Zulu", "[[Missing]] then [[ Alpha  Note |read this]]"),
            unit("a", "Alpha Note", "No refs"),
            unit("b", "Beta", "[[missing]] and [[missing|again]]", {"see": "[[unit-c]]"}),
            unit("unit-c", "Gamma", "No refs"),
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "b",
            "title": "Beta",
            "reference": "missing",
            "normalized_reference": "missing",
            "reference_count": "2",
            "matched_candidate_count": "0",
        },
        {
            "unit_id": "z",
            "title": "Zulu",
            "reference": "Missing",
            "normalized_reference": "missing",
            "reference_count": "1",
            "matched_candidate_count": "0",
        },
    ]


def test_unit_unresolved_reference_csv_accepts_mappings_and_path_mode(tmp_path):
    units = [
        {"id": "known-id", "title": "Known Title", "content": ""},
        {
            "id": "map",
            "title": "Mapping",
            "content": "[[known title]] [[known-id]] [[Unknown]]",
            "metadata": {"nested": ["[[Another Missing]]"]},
        },
    ]
    path = tmp_path / "reports" / "unresolved.csv"

    expected = export_unit_unresolved_reference_csv(units)
    stats = export_unit_unresolved_reference_csv(units, path)

    assert rows(expected) == [
        {
            "unit_id": "map",
            "title": "Mapping",
            "reference": "Another Missing",
            "normalized_reference": "another missing",
            "reference_count": "1",
            "matched_candidate_count": "0",
        },
        {
            "unit_id": "map",
            "title": "Mapping",
            "reference": "Unknown",
            "normalized_reference": "unknown",
            "reference_count": "1",
            "matched_candidate_count": "0",
        },
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }
