from __future__ import annotations

import csv
from io import StringIO

from graph.export.edge_annotation_inventory_csv import export_edge_annotation_inventory_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    from_unit_id: str = "from",
    to_unit_id: str = "to",
    relation: EdgeRelation | str | None = EdgeRelation.RELATES_TO,
    source: EdgeSource | str | None = EdgeSource.INFERRED,
    metadata: object = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        weight=1.0,
        metadata=metadata if metadata is not None else {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_annotation_inventory_empty_input_has_header_only():
    assert export_edge_annotation_inventory_csv([]) == (
        "edge_id,from_unit_id,to_unit_id,relation,source,source_project,provenance,"
        "annotation_keys,note_comment_length,evidence_count,reference_count,has_rationale_text\n"
    )


def test_edge_annotation_inventory_exports_one_row_per_edge_with_annotations():
    text = export_edge_annotation_inventory_csv(
        [
            edge(
                "edge-b",
                from_unit_id="unit-b",
                to_unit_id="unit-c",
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                metadata={
                    "source_project": "max",
                    "provenance": "import:1",
                    "note": "Short note",
                    "comment": ["first", "second"],
                    "rationale": "Because it cites the source.",
                    "evidence": ["quote one", "quote two"],
                    "references": ["ref-a"],
                    "citations": ["cite-a", "cite-b"],
                },
            ),
            edge("edge-a", from_unit_id="unit-a", to_unit_id="unit-b", relation="custom", source=None),
        ]
    )

    assert rows(text) == [
        {
            "edge_id": "edge-a",
            "from_unit_id": "unit-a",
            "to_unit_id": "unit-b",
            "relation": "custom",
            "source": "Unknown",
            "source_project": "",
            "provenance": "",
            "annotation_keys": "",
            "note_comment_length": "0",
            "evidence_count": "0",
            "reference_count": "0",
            "has_rationale_text": "false",
        },
        {
            "edge_id": "edge-b",
            "from_unit_id": "unit-b",
            "to_unit_id": "unit-c",
            "relation": "references",
            "source": "source",
            "source_project": "max",
            "provenance": "import:1",
            "annotation_keys": "citations; comment; evidence; note; rationale; references",
            "note_comment_length": "21",
            "evidence_count": "2",
            "reference_count": "3",
            "has_rationale_text": "true",
        },
    ]


def test_edge_annotation_inventory_handles_scalar_list_and_mapping_values():
    text = export_edge_annotation_inventory_csv(
        [
            edge(
                "a",
                metadata={
                    "project": "project-a",
                    "source_provenance": ["rule", "manual"],
                    "notes": ["one", None, "two"],
                    "evidence": {"quote": "quoted text", "empty": ""},
                    "citations": "doi:10/example",
                    "rationale": ["", "   "],
                },
            )
        ]
    )

    assert rows(text)[0] == {
        "edge_id": "a",
        "from_unit_id": "from",
        "to_unit_id": "to",
        "relation": "relates_to",
        "source": "inferred",
        "source_project": "project-a",
        "provenance": "rule; manual",
        "annotation_keys": "citations; evidence; notes",
        "note_comment_length": "6",
        "evidence_count": "1",
        "reference_count": "1",
        "has_rationale_text": "false",
    }


def test_edge_annotation_inventory_handles_non_mapping_metadata():
    text = export_edge_annotation_inventory_csv([edge("a", metadata=["not", "a", "mapping"])])

    assert rows(text)[0]["annotation_keys"] == ""
    assert rows(text)[0]["note_comment_length"] == "0"
    assert rows(text)[0]["has_rationale_text"] == "false"


def test_edge_annotation_inventory_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "edge-annotations.csv"
    edges = [edge("a", metadata={"note": "note"})]

    expected = export_edge_annotation_inventory_csv(edges)
    stats = export_edge_annotation_inventory_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_edge_annotation_inventory_is_deterministic_for_reversed_input():
    edges = [
        edge("b", from_unit_id="unit-b", to_unit_id="unit-c", metadata={"note": "b"}),
        edge("a", from_unit_id="unit-a", to_unit_id="unit-b", metadata={"note": "a"}),
    ]

    assert export_edge_annotation_inventory_csv(edges) == export_edge_annotation_inventory_csv(reversed(edges))
