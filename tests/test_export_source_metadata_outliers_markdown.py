from __future__ import annotations

from graph.export.source_metadata_outliers_markdown import export_source_metadata_outliers_markdown
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def _unit(index: int, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=SourceProject.SPOTIFY_STREAMING_HISTORY,
        source_id=f"unit-{index}",
        source_entity_type="play",
        title=f"Unit {index}",
        content="content",
        metadata=metadata,
    )


def test_export_source_metadata_outliers_markdown_reports_rare_keys():
    units = [
        _unit(1, {"common": "yes", "rare": "alpha"}),
        _unit(2, {"common": "yes"}),
        _unit(3, {"common": "yes"}),
        _unit(4, {"common": "yes"}),
    ]

    markdown = export_source_metadata_outliers_markdown(units, max_key_frequency=0.25)

    assert "# Source Metadata Outliers" in markdown
    assert "spotify_streaming_history / play" in markdown
    assert "| rare | 1 | alpha (1) | Unit 1 |" in markdown
    assert "common" not in markdown


def test_export_source_metadata_outliers_markdown_path_mode(tmp_path):
    path = tmp_path / "outliers.md"
    units = [_unit(1, {"rare": "alpha"}), _unit(2, {})]

    stats = export_source_metadata_outliers_markdown(units, path, max_key_frequency=0.5)

    assert stats["path"] == str(path)
    assert stats["units_scanned"] == 2
    assert stats["groups_reported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
    assert path.read_text(encoding="utf-8").startswith("# Source Metadata Outliers")
