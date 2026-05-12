from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.relation_weight_distribution_csv import export_relation_weight_distribution_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    relation: EdgeRelation | str | None = EdgeRelation.RELATES_TO,
    source: EdgeSource | str | None = EdgeSource.INFERRED,
    weight: object = 1.0,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=source,
        weight=weight,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_weight_distribution_csv_empty_input_has_header_only():
    assert export_relation_weight_distribution_csv([]) == (
        "relation,source,edge_count,min_weight,max_weight,average_weight,zero_weight_count,"
        "negative_weight_count,strong_edge_count\n"
    )


def test_relation_weight_distribution_csv_groups_by_relation_and_source():
    text = export_relation_weight_distribution_csv(
        [
            edge("a", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, weight=2),
            edge("b", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, weight=0),
            edge("c", relation=EdgeRelation.REFERENCES, source=EdgeSource.MANUAL, weight=-1),
            edge("d", relation="custom", source="external", weight=0.5),
        ],
        strong_threshold=1.0,
    )

    assert rows(text) == [
        {
            "relation": "custom",
            "source": "external",
            "edge_count": "1",
            "min_weight": "0.50",
            "max_weight": "0.50",
            "average_weight": "0.50",
            "zero_weight_count": "0",
            "negative_weight_count": "0",
            "strong_edge_count": "0",
        },
        {
            "relation": "references",
            "source": "manual",
            "edge_count": "1",
            "min_weight": "-1.00",
            "max_weight": "-1.00",
            "average_weight": "-1.00",
            "zero_weight_count": "0",
            "negative_weight_count": "1",
            "strong_edge_count": "0",
        },
        {
            "relation": "references",
            "source": "source",
            "edge_count": "2",
            "min_weight": "0.00",
            "max_weight": "2.00",
            "average_weight": "1.00",
            "zero_weight_count": "1",
            "negative_weight_count": "0",
            "strong_edge_count": "1",
        },
    ]


def test_relation_weight_distribution_csv_uses_custom_strong_threshold():
    text = export_relation_weight_distribution_csv(
        [
            edge("a", weight=0.75),
            edge("b", weight=1.0),
            edge("c", weight=1.25),
        ],
        strong_threshold=0.75,
    )

    assert rows(text)[0]["strong_edge_count"] == "3"


def test_relation_weight_distribution_csv_formats_nonnumeric_weights_as_zero():
    text = export_relation_weight_distribution_csv([edge("a", weight="heavy")])

    assert rows(text)[0]["min_weight"] == "0.00"
    assert rows(text)[0]["zero_weight_count"] == "1"


def test_relation_weight_distribution_csv_validates_strong_threshold():
    with pytest.raises(ValueError, match="strong_threshold"):
        export_relation_weight_distribution_csv([], strong_threshold="high")
    with pytest.raises(ValueError, match="strong_threshold"):
        export_relation_weight_distribution_csv([], strong_threshold=True)


def test_relation_weight_distribution_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "relation-weight-distribution.csv"
    edges = [edge("a", weight=2.0)]

    expected = export_relation_weight_distribution_csv(edges, strong_threshold=1.5)
    stats = export_relation_weight_distribution_csv(edges, path, strong_threshold=1.5)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "rows_exported": 1,
        "strong_threshold": 1.5,
        "bytes_written": path.stat().st_size,
    }
