from __future__ import annotations

import csv
from io import StringIO

from graph.export.edge_metadata_schema_csv import export_edge_metadata_schema_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    relation: EdgeRelation | str | None = EdgeRelation.RELATES_TO,
    source: EdgeSource | str | None = EdgeSource.INFERRED,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=source,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_metadata_schema_csv_empty_input_has_header_only():
    assert export_edge_metadata_schema_csv([]) == (
        "relation,source,metadata_key,edge_count,populated_edge_count,coverage_percent,"
        "observed_type_names,example_values\n"
    )


def test_edge_metadata_schema_csv_groups_keys_by_relation_and_source():
    text = export_edge_metadata_schema_csv(
        [
            edge("a", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, metadata={"kind": "citation"}),
            edge("b", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, metadata={"kind": "link"}),
            edge("c", relation=EdgeRelation.REFERENCES, source=EdgeSource.MANUAL, metadata={"kind": "curated"}),
            edge("d", relation="custom", source="external", metadata={"kind": "imported"}),
        ]
    )

    assert rows(text) == [
        {
            "relation": "custom",
            "source": "external",
            "metadata_key": "kind",
            "edge_count": "1",
            "populated_edge_count": "1",
            "coverage_percent": "100.00",
            "observed_type_names": "str",
            "example_values": "imported",
        },
        {
            "relation": "references",
            "source": "manual",
            "metadata_key": "kind",
            "edge_count": "1",
            "populated_edge_count": "1",
            "coverage_percent": "100.00",
            "observed_type_names": "str",
            "example_values": "curated",
        },
        {
            "relation": "references",
            "source": "source",
            "metadata_key": "kind",
            "edge_count": "2",
            "populated_edge_count": "2",
            "coverage_percent": "100.00",
            "observed_type_names": "str",
            "example_values": "citation; link",
        },
    ]


def test_edge_metadata_schema_csv_coverage_uses_populated_values_in_group():
    text = export_edge_metadata_schema_csv(
        [
            edge("a", metadata={"confidence_note": "strong", "empty": ""}),
            edge("b", metadata={"confidence_note": "", "empty": []}),
            edge("c", metadata={"confidence_note": None}),
            edge("d", metadata={}),
        ]
    )

    by_key = {row["metadata_key"]: row for row in rows(text)}
    assert by_key["confidence_note"] == {
        "relation": "relates_to",
        "source": "inferred",
        "metadata_key": "confidence_note",
        "edge_count": "4",
        "populated_edge_count": "1",
        "coverage_percent": "25.00",
        "observed_type_names": "null; str",
        "example_values": "strong",
    }
    assert by_key["empty"]["coverage_percent"] == "0.00"
    assert by_key["empty"]["observed_type_names"] == "list; str"


def test_edge_metadata_schema_csv_limits_examples_deterministically():
    text = export_edge_metadata_schema_csv(
        [
            edge("a", metadata={"label": "zeta", "payload": {"b": 2, "a": 1}}),
            edge("b", metadata={"label": "alpha", "payload": {"a": 1, "b": 2}}),
            edge("c", metadata={"label": "beta", "payload": ["x", "y"]}),
            edge("d", metadata={"label": "gamma", "payload": {"c": 3}}),
        ]
    )

    by_key = {row["metadata_key"]: row for row in rows(text)}
    assert by_key["label"]["example_values"] == "alpha; beta; gamma"
    assert by_key["payload"]["observed_type_names"] == "dict; list"
    assert export_edge_metadata_schema_csv(
        [
            edge("d", metadata={"label": "gamma", "payload": {"c": 3}}),
            edge("c", metadata={"label": "beta", "payload": ["x", "y"]}),
            edge("b", metadata={"label": "alpha", "payload": {"a": 1, "b": 2}}),
            edge("a", metadata={"label": "zeta", "payload": {"b": 2, "a": 1}}),
        ]
    ) == text


def test_edge_metadata_schema_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "edge-metadata-schema.csv"
    edges = [edge("a", metadata={"kind": "citation"})]

    expected = export_edge_metadata_schema_csv(edges)
    stats = export_edge_metadata_schema_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "schema_key_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
