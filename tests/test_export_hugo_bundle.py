from __future__ import annotations

from datetime import datetime, timezone

import yaml

from graph.export import export_units_to_hugo_bundle
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str, *, title: str, source_id: str | None = None, metadata: dict | None = None):
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"# Body for {title}\n",
        content_type=ContentType.INSIGHT,
        tags=["zeta", "alpha"],
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def frontmatter(path):
    text = path.read_text(encoding="utf-8")
    _, yaml_text, body = text.split("---\n", 2)
    return yaml.safe_load(yaml_text), body


def test_export_units_to_hugo_bundle_writes_markdown_files(tmp_path):
    stats = export_units_to_hugo_bundle(
        [unit("unit-a", title="Alpha Note", metadata={"external_url": "https://example.test"})],
        tmp_path,
    )

    metadata, body = frontmatter(tmp_path / "alpha-note.md")
    assert metadata["title"] == "Alpha Note"
    assert metadata["date"] == "2026-05-01T10:15:00+00:00"
    assert metadata["lastmod"] == "2026-05-01T10:15:00+00:00"
    assert metadata["tags"] == ["alpha", "zeta"]
    assert metadata["source_project"] == "max"
    assert metadata["source_id"] == "source-unit-a"
    assert metadata["external_url"] == "https://example.test"
    assert body == "\n# Body for Alpha Note\n\n"
    assert stats["files_written"] == 1


def test_export_units_to_hugo_bundle_resolves_slug_collisions_and_index(tmp_path):
    export_units_to_hugo_bundle(
        [
            unit("unit-b", title="Same Title", source_id="beta"),
            unit("unit-a", title="Same Title", source_id="alpha"),
        ],
        tmp_path,
        include_index=True,
    )

    assert (tmp_path / "same-title-beta.md").exists()
    assert (tmp_path / "same-title-alpha.md").exists()
    index = (tmp_path / "_index.md").read_text(encoding="utf-8")
    assert "## max" in index
    assert "- Same Title (alpha)" in index
    assert "- Same Title (beta)" in index
