from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_url_quality_csv
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, *, metadata: dict | None = None, content: str = "", source_project: str = "Project") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Title {unit_id}",
        content=content,
        metadata=metadata or {},
        tags=[],
        created_at=None,
        updated_at=None,
        ingested_at=None,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_url_quality_csv_empty_input_has_header_only():
    assert export_unit_url_quality_csv([]) == "source,unit_id,title,url,issue,detail\n"


def test_unit_url_quality_csv_detects_metadata_and_content_issues_deterministically():
    long_url = "https://example.test/" + ("a" * 2050)
    text = export_unit_url_quality_csv(
        [
            unit(
                "a",
                metadata={"url": "example.test/page", "external_url": " ftp://example.test/file "},
                content=f"See https://badhost and {long_url}",
            )
        ]
    )

    assert rows(text) == [
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "url": "example.test/page",
            "issue": "missing_scheme",
            "detail": "URL has no scheme",
        },
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "url": " ftp://example.test/file ",
            "issue": "unsupported_scheme",
            "detail": "ftp",
        },
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "url": " ftp://example.test/file ",
            "issue": "whitespace",
            "detail": "URL contains leading, trailing, or embedded whitespace",
        },
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "url": "https://badhost",
            "issue": "malformed",
            "detail": "URL host does not look fully qualified",
        },
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "url": long_url,
            "issue": "suspiciously_long",
            "detail": "2071 characters",
        },
    ]


def test_unit_url_quality_csv_reports_duplicate_urls_per_unit_once():
    text = export_unit_url_quality_csv([unit("a", metadata={"url": "https://example.test/a"}, content="https://example.test/a")])

    assert rows(text) == [
        {
            "source": "Project",
            "unit_id": "a",
            "title": "Title a",
            "url": "https://example.test/a",
            "issue": "duplicate",
            "detail": "2 occurrences in unit",
        }
    ]


def test_unit_url_quality_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "urls.csv"
    units = [unit("a", metadata={"url": "example.test"})]

    expected = export_unit_url_quality_csv(units)
    stats = export_unit_url_quality_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {"path": str(path), "unit_count": 1, "rows_exported": 1, "bytes_written": path.stat().st_size}
