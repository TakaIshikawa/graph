from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_relation_confidence_buckets_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    relation: EdgeRelation | str | None,
    confidence: object = None,
    *,
    direct_confidence: object = None,
) -> KnowledgeEdge:
    edge = KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=1.0,
        metadata={} if confidence is None else {"confidence": confidence},
    )
    if direct_confidence is not None:
        object.__setattr__(edge, "confidence", direct_confidence)
    return edge


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_confidence_buckets_csv_buckets_confidence_by_relation():
    text = export_relation_confidence_buckets_csv(
        [
            edge("a", EdgeRelation.REFERENCES, 0.0),
            edge("b", EdgeRelation.REFERENCES, 0.24),
            edge("c", EdgeRelation.REFERENCES, 0.25),
            edge("d", EdgeRelation.REFERENCES, 1.0),
            edge("e", EdgeRelation.REFERENCES, None),
        ]
    )

    assert text.splitlines()[0] == (
        "relation,bucket,edge_count,average_confidence,unknown_confidence_count"
    )
    assert rows(text) == [
        {
            "relation": "references",
            "bucket": "0.00-0.25",
            "edge_count": "2",
            "average_confidence": "0.12",
            "unknown_confidence_count": "1",
        },
        {
            "relation": "references",
            "bucket": "0.25-0.50",
            "edge_count": "1",
            "average_confidence": "0.25",
            "unknown_confidence_count": "1",
        },
        {
            "relation": "references",
            "bucket": "0.75-1.00",
            "edge_count": "1",
            "average_confidence": "1.00",
            "unknown_confidence_count": "1",
        },
    ]


def test_relation_confidence_buckets_csv_prefers_edge_confidence_over_metadata():
    text = export_relation_confidence_buckets_csv(
        [edge("a", "related", 0.9, direct_confidence=0.1)]
    )

    assert rows(text) == [
        {
            "relation": "related",
            "bucket": "0.00-0.25",
            "edge_count": "1",
            "average_confidence": "0.10",
            "unknown_confidence_count": "0",
        }
    ]


def test_relation_confidence_buckets_csv_treats_invalid_values_as_unknown():
    text = export_relation_confidence_buckets_csv(
        [
            edge("a", "related", True),
            edge("b", "related", -0.1),
            edge("c", "related", 1.1),
            edge("d", None, "high"),
        ]
    )

    assert rows(text) == [
        {
            "relation": "related",
            "bucket": "Unknown",
            "edge_count": "0",
            "average_confidence": "",
            "unknown_confidence_count": "3",
        },
        {
            "relation": "Unknown",
            "bucket": "Unknown",
            "edge_count": "0",
            "average_confidence": "",
            "unknown_confidence_count": "1",
        },
    ]


def test_relation_confidence_buckets_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "relation-confidence-buckets.csv"
    edges = [edge("a", EdgeRelation.RELATES_TO, 0.7)]

    expected = export_relation_confidence_buckets_csv(edges)
    stats = export_relation_confidence_buckets_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "relation_count": 1,
        "rows_exported": 1,
        "bucket_size": 0.25,
        "bytes_written": path.stat().st_size,
    }


@pytest.mark.parametrize("bucket_size", [0, -0.1, 1.1, True, "0.25"])
def test_relation_confidence_buckets_csv_validates_bucket_size(bucket_size):
    with pytest.raises(ValueError, match="bucket_size must be a positive number not greater than 1"):
        export_relation_confidence_buckets_csv([], bucket_size=bucket_size)
