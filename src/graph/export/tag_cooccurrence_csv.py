"""CSV export helpers for tag co-occurrence reports."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from io import StringIO

from graph.rag import build_tag_cooccurrence_matrix
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["tag_a", "tag_b", "count"]


def export_tag_cooccurrence_csv(
    units: Iterable[KnowledgeUnit],
    *,
    min_count: int = 1,
) -> str:
    """Return tag co-occurrence counts as CSV."""
    matrix = build_tag_cooccurrence_matrix(units, min_count=min_count)
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()

    rows = sorted(
        (
            {
                "tag_a": pair["source"],
                "tag_b": pair["target"],
                "count": pair["count"],
            }
            for pair in matrix["pairs"]
        ),
        key=lambda row: (_sort_key(row["tag_a"]), _sort_key(row["tag_b"])),
    )
    writer.writerows(rows)

    return output.getvalue()


def _sort_key(value: object) -> tuple[str, str]:
    text = str(value or "")
    return (text.casefold(), text)
