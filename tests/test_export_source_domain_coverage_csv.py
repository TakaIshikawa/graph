from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_source_domain_coverage_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, tzinfo=timezone.utc)


def unit(unit_id: str, *, metadata: dict | None = None, source_entity_type: str = "item") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type=source_entity_type,
        title=f"Title {unit_id}",
        content="Content",
        metadata=metadata or {},
        tags=[],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_source_domain_coverage_csv_empty_input_returns_header():
    assert export_source_domain_coverage_csv([]) == "domain,source_type,source_label,unit_count\n"


def test_export_source_domain_coverage_csv_groups_repeated_domains_once_per_unit():
    text = export_source_domain_coverage_csv(
        [
            unit("a", metadata={"url": "https://Example.com/a", "links": ["https://example.com/b"]}),
            unit("b", metadata={"source_url": "https://example.com/c"}),
            unit("c", metadata={"url": "https://other.test/path"}),
        ]
    )

    assert rows(text) == [
        {"domain": "example.com", "source_type": "item", "source_label": "max", "unit_count": "2"},
        {"domain": "other.test", "source_type": "item", "source_label": "max", "unit_count": "1"},
    ]


def test_export_source_domain_coverage_csv_uses_source_type_and_label_metadata():
    text = export_source_domain_coverage_csv(
        [
            unit(
                "a",
                metadata={
                    "source": {"url": "https://example.test/a", "label": "Example"},
                    "source_type": "article",
                },
            )
        ]
    )

    assert rows(text)[0] == {
        "domain": "example.test",
        "source_type": "article",
        "source_label": "Example",
        "unit_count": "1",
    }


def test_export_source_domain_coverage_csv_keeps_missing_and_invalid_urls_unknown():
    text = export_source_domain_coverage_csv(
        [
            unit("missing", metadata={}),
            unit("invalid", metadata={"url": "not a url"}),
        ]
    )

    assert rows(text) == [
        {"domain": "unknown", "source_type": "item", "source_label": "max", "unit_count": "2"},
    ]


def test_export_source_domain_coverage_csv_path_mode(tmp_path):
    path = tmp_path / "domains.csv"
    stats = export_source_domain_coverage_csv([unit("a", metadata={"url": "https://example.test"})], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["domain"] == "example.test"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
