from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.hacker_news_saved import HackerNewsSavedAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_hacker_news_saved_imports_story_metadata(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 424242,
                    "title": "Interesting systems post",
                    "url": "https://example.com/systems",
                    "by": "pg",
                    "time": 1735689600,
                    "score": 123,
                    "type": "story",
                    "text": "A short note from the export.",
                    "kids": [1, 2, 3],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["saved_item"])

    saved_items = [unit for unit in result.units if unit.source_entity_type == "saved_item"]
    assert len(saved_items) == 1
    unit = saved_items[0]
    assert unit.source_project == SourceProject.HACKER_NEWS_SAVED
    assert unit.source_entity_type == "saved_item"
    assert unit.source_id == "hacker_news_saved:424242"
    assert unit.title == "Interesting systems post"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert "A short note from the export." in unit.content
    assert "URL: https://example.com/systems" in unit.content
    assert "Hacker News: https://news.ycombinator.com/item?id=424242" in unit.content
    assert unit.metadata["item_id"] == 424242
    assert unit.metadata["hn_item_id"] == 424242
    assert unit.metadata["author"] == "pg"
    assert unit.metadata["score"] == 123
    assert unit.metadata["item_type"] == "story"
    assert unit.metadata["hn_item_type"] == "story"
    assert unit.metadata["comment_count"] == 3
    assert unit.metadata["time"] == 1735689600
    assert unit.metadata["time_iso"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["external_url"] == "https://example.com/systems"
    assert unit.metadata["source_url"] == "https://example.com/systems"
    assert unit.metadata["hn_item_url"] == "https://news.ycombinator.com/item?id=424242"


def test_hacker_news_saved_normalizes_comment_metadata_and_references(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 424243,
                    "type": "comment",
                    "by": "dang",
                    "text": "A saved comment.",
                    "parent": 424242,
                    "story_id": 424200,
                    "time": 1735689600,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["saved_item"])

    saved_items = [unit for unit in result.units if unit.source_entity_type == "saved_item"]
    assert len(saved_items) == 1
    unit = saved_items[0]
    assert unit.metadata["hn_item_type"] == "comment"
    assert unit.metadata["hn_item_id"] == 424243
    assert unit.metadata["hn_parent_id"] == 424242
    assert unit.metadata["hn_story_id"] == 424200
    assert unit.tags == ["hacker_news", "comment"]


def test_hacker_news_saved_preserves_sparse_unknown_items(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps([{"id": 999, "type": "pollopt", "text": "Sparse saved item."}]),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["hn_item_id"] == 999
    assert result.units[0].metadata["hn_item_type"] == "unknown"


def test_hacker_news_saved_accepts_top_level_items_and_saved_items(tmp_path):
    items_path = tmp_path / "items.json"
    saved_items_path = tmp_path / "saved_items.json"
    items_path.write_text(json.dumps({"items": [{"id": 1, "title": "Items shape", "time": 1735689600}]}), encoding="utf-8")
    saved_items_path.write_text(
        json.dumps({"saved_items": [{"id": 2, "title": "Saved items shape", "time": 1735689601}]}),
        encoding="utf-8",
    )

    items_result = HackerNewsSavedAdapter(path=str(items_path)).ingest()
    saved_items_result = HackerNewsSavedAdapter(path=str(saved_items_path)).ingest()

    assert [unit.title for unit in items_result.units] == ["Items shape"]
    assert [unit.title for unit in saved_items_result.units] == ["Saved items shape"]


def test_hacker_news_saved_keeps_url_only_items_and_uses_hn_source_url_fallback(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {"id": 100, "url": "https://example.com/only-url", "time": 1735689600},
                {"id": 101, "title": "Ask HN item", "type": "story", "time": 1735689601},
            ]
        ),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["saved_item"])

    assert [unit.title for unit in result.units] == ["https://example.com/only-url", "Ask HN item"]
    assert result.units[0].metadata["external_url"] == "https://example.com/only-url"
    assert result.units[1].metadata["source_url"] == "https://news.ycombinator.com/item?id=101"
    assert "external_url" not in result.units[1].metadata


def test_hacker_news_saved_filters_by_sync_state_and_entity_type(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {"id": 1, "title": "Old", "time": 1735689600},
                {"id": 2, "title": "Boundary", "time": 1735689601},
                {"id": 3, "title": "New", "time": 1735689602},
            ]
        ),
        encoding="utf-8",
    )

    skipped = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["comment"])
    result = HackerNewsSavedAdapter(path=str(path)).ingest(
        since=SyncState(
            source_project="hacker_news_saved",
            source_entity_type="saved_item",
            last_sync_at=datetime.fromtimestamp(1735689601, tz=timezone.utc),
        )
    )

    assert skipped.units == []
    assert skipped.edges == []
    assert [unit.title for unit in result.units] == ["New"]


def test_hacker_news_saved_adapter_is_registered():
    assert "hacker_news_saved" in list_adapters()
    adapter = get_adapter("hacker-news-saved", path="/tmp/saved.json")
    assert isinstance(adapter, HackerNewsSavedAdapter)
    assert adapter.name == "hacker_news_saved"
    assert adapter.entity_types == ["saved_item", "submitter", "domain"]


def test_hacker_news_saved_emits_submitter_units_and_edges(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "title": "One",
                    "url": "https://www.example.com/one",
                    "author": "PG",
                    "type": "story",
                    "time": 1735689600,
                },
                {
                    "id": 2,
                    "title": "Two",
                    "url": "https://example.com/two",
                    "by": " pg ",
                    "type": "comment",
                    "time": 1735689601,
                },
                {
                    "id": 3,
                    "title": "Three",
                    "url": "https://news.ycombinator.com/item?id=3",
                    "submitter": "dang",
                    "type": "story",
                    "time": 1735689602,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["saved_item", "submitter"])

    saved_items = [unit for unit in result.units if unit.source_entity_type == "saved_item"]
    submitters = sorted((unit for unit in result.units if unit.source_entity_type == "submitter"), key=lambda unit: unit.title)
    assert [unit.title for unit in submitters] == ["PG", "dang"]
    pg = next(unit for unit in submitters if unit.title == "PG")
    assert pg.metadata["submitter"] == "PG"
    assert pg.metadata["item_count"] == 2
    assert pg.metadata["item_source_ids"] == sorted(item.source_id for item in saved_items if item.metadata["submitter"].strip().casefold() == "pg")
    assert pg.metadata["item_types"] == ["comment", "story"]
    assert pg.metadata["domains"] == ["example.com"]
    assert pg.metadata["source_files"] == ["saved.json"]
    assert len(result.edges) == 3
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}
    assert {edge.to_unit_id for edge in result.edges} == {item.source_id for item in saved_items}


def test_hacker_news_saved_submitter_filtering(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(json.dumps([{"id": 1, "title": "One", "by": "pg", "time": 1735689600}]), encoding="utf-8")

    submitter_only = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["submitter"])
    item_only = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["saved_item"])

    assert [unit.source_entity_type for unit in submitter_only.units] == ["submitter"]
    assert submitter_only.edges == []
    assert [unit.source_entity_type for unit in item_only.units] == ["saved_item"]
    assert item_only.edges == []


def test_hacker_news_saved_emits_domain_units_and_edges(tmp_path):
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "title": "One",
                    "url": "https://www.example.com/one",
                    "by": "pg",
                    "type": "story",
                    "time": 1735689600,
                },
                {
                    "id": 2,
                    "title": "Two",
                    "url": "https://example.com/two",
                    "by": "dang",
                    "type": "comment",
                    "time": 1735689601,
                },
                {
                    "id": 3,
                    "title": "Three",
                    "url": "not a valid url",
                    "by": "pg",
                    "type": "story",
                    "time": 1735689602,
                },
                {
                    "id": 4,
                    "title": "Four",
                    "by": "pg",
                    "type": "story",
                    "time": 1735689603,
                },
            ]
        ),
        encoding="utf-8",
    )

    domains_only = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["domain"])
    combined = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["saved_item", "domain"])
    items_only = HackerNewsSavedAdapter(path=str(path)).ingest(entity_types=["saved_item"])

    domains = {unit.metadata["domain"]: unit for unit in domains_only.units}
    assert list(domains) == ["example.com"]
    domain = domains["example.com"]
    assert domain.source_entity_type == "domain"
    assert domain.source_id.startswith("hacker_news_saved:domain:")
    assert domain.metadata["item_count"] == 2
    assert domain.metadata["item_source_ids"] == ["hacker_news_saved:1", "hacker_news_saved:2"]
    assert domain.metadata["item_types"] == ["comment", "story"]
    assert domain.metadata["submitters"] == ["dang", "pg"]
    assert domains_only.edges == []

    domain_edges = [edge for edge in combined.edges if edge.metadata["to_entity_type"] == "domain"]
    assert len(domain_edges) == 2
    assert {edge.from_unit_id for edge in domain_edges} == {"hacker_news_saved:1", "hacker_news_saved:2"}
    assert {edge.to_unit_id for edge in domain_edges} == {domain.source_id}
    assert {edge.relation for edge in domain_edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in domain_edges} == {EdgeSource.SOURCE}
    assert all(edge.metadata["domain"] == "example.com" for edge in domain_edges)

    assert {unit.source_entity_type for unit in items_only.units} == {"saved_item"}
    assert items_only.edges == []
