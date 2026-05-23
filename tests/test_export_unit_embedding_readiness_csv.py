from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_embedding_readiness_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_embedding_readiness_empty_input_has_header():
    assert export_unit_embedding_readiness_csv([]) == (
        "unit_id,title,content_length,has_language,has_source,has_sensitive_flags,readiness_score,blockers\n"
    )


def test_unit_embedding_readiness_scores_ready_unit():
    result = rows(
        export_unit_embedding_readiness_csv(
            [
                {
                    "id": "u1",
                    "title": "Ready",
                    "content": "hello world",
                    "source_id": "s1",
                    "metadata": {"language": "en"},
                }
            ]
        )
    )[0]

    assert result["content_length"] == "11"
    assert result["has_language"] == "true"
    assert result["has_source"] == "true"
    assert result["readiness_score"] == "100"
    assert result["blockers"] == ""


def test_unit_embedding_readiness_reports_blockers_and_path_mode(tmp_path):
    path = tmp_path / "readiness.csv"
    stats = export_unit_embedding_readiness_csv(
        [
            {
                "id": "bad",
                "title": "Bad",
                "content": "",
                "metadata": {"sensitive_flags": ["pii"], "duplicate": True},
            }
        ],
        path,
    )

    result = rows(path.read_text(encoding="utf-8"))[0]
    assert result["has_sensitive_flags"] == "true"
    assert result["readiness_score"] == "0"
    assert result["blockers"] == "empty_content;sensitive_flags;missing_source;duplicate_content"
    assert stats["unit_count"] == 1
