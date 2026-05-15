from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from graph.adapters.browser_history_csv import BrowserHistoryCsvAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_browser_history_csv_ingests_common_columns(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "URL,Title,Visit Time,Visit Count,Last-Visit-Time,typed_count\n"
        "https://Example.com:443/research?q=graph#fragment,Graph Research,"
        "2025-04-24T12:00:00Z,7,2025-04-25T13:30:00Z,2\n",
        encoding="utf-8",
    )

    result = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["web_history"])

    assert len(result.units) == 1
    unit = result.units[0]
    normalized_url = "https://example.com/research?q=graph"
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
    assert unit.source_project == SourceProject.BROWSER_HISTORY_CSV
    assert unit.source_entity_type == "web_history"
    assert unit.source_id == f"browser_history_csv:{digest}"
    assert unit.title == "Graph Research"
    assert unit.content == f"Graph Research\nURL: {normalized_url}"
    assert unit.content_type == ContentType.METADATA
    assert unit.created_at == datetime(2025, 4, 24, 12, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 4, 25, 13, 30, tzinfo=timezone.utc)
    assert unit.metadata == {
        "url": "https://Example.com:443/research?q=graph#fragment",
        "normalized_url": normalized_url,
        "domain": "example.com",
        "visit_time": "2025-04-24T12:00:00Z",
        "last_visit_time": "2025-04-25T13:30:00Z",
        "visit_timestamps": [
            "2025-04-24T12:00:00+00:00",
            "2025-04-25T13:30:00+00:00",
        ],
        "visit_count": 7,
        "typed_count": 2,
        "source_file": "history.csv",
    }


def test_browser_history_csv_tolerates_separator_and_case_variants(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "Url,Page Title,lastVisitTime,Visit_Count,Typed Count\n"
        "example.org/docs/page,,13386556800000000,3,1\n",
        encoding="utf-8",
    )

    result = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["web_history"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "example.org/docs/page"
    assert unit.metadata["normalized_url"] == "https://example.org/docs/page"
    assert unit.metadata["domain"] == "example.org"
    assert unit.metadata["visit_count"] == 3
    assert unit.metadata["typed_count"] == 1
    assert unit.metadata["last_visit_time"] == "13386556800000000"
    assert unit.updated_at == datetime(2025, 3, 16, tzinfo=timezone.utc)


def test_browser_history_csv_deduplicates_by_normalized_url(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "url,title,last_visit_time\n"
        "https://example.com/,Home,2025-04-24T12:00:00Z\n"
        "https://EXAMPLE.com/#section,Duplicate,2025-04-25T12:00:00Z\n",
        encoding="utf-8",
    )

    result = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["web_history"])

    assert len(result.units) == 1
    assert result.units[0].title == "Home"
    assert result.units[0].metadata["normalized_url"] == "https://example.com/"


def test_browser_history_csv_emits_domain_aggregates_and_edges(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "url,title,last_visit_time\n"
        "https://example.com/,Home,2025-04-24T12:00:00Z\n"
        "https://example.com/docs,Docs,2025-04-25T12:00:00Z\n"
        "https://other.example/path,Other,2025-04-26T12:00:00Z\n",
        encoding="utf-8",
    )

    result = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest()

    domains = [unit for unit in result.units if unit.source_entity_type == "domain"]
    visits = [unit for unit in result.units if unit.source_entity_type == "web_history"]
    assert len(domains) == 2
    assert len(visits) == 3

    example = next(unit for unit in domains if unit.metadata["domain"] == "example.com")
    example_visits = [unit for unit in visits if unit.metadata["domain"] == "example.com"]
    assert example.source_id.startswith("browser_history_csv:domain:")
    assert example.title == "example.com"
    assert example.metadata["visit_count"] == 2
    assert example.metadata["page_source_ids"] == sorted(unit.source_id for unit in example_visits)
    assert example.metadata["normalized_urls"] == [
        "https://example.com/",
        "https://example.com/docs",
    ]

    domain_edges = [edge for edge in result.edges if edge.to_unit_id == example.source_id]
    assert {edge.from_unit_id for edge in domain_edges} == {unit.source_id for unit in example_visits}
    assert all(edge.relation == EdgeRelation.RELATES_TO for edge in domain_edges)
    assert all(edge.source == EdgeSource.SOURCE for edge in domain_edges)
    assert all(edge.metadata["relation_type"] == "visit_domain" for edge in domain_edges)


def test_browser_history_csv_domain_filtering_and_unparseable_urls(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "url,title,last_visit_time\n"
        "https://example.com/,Home,2025-04-24T12:00:00Z\n"
        "file:///Users/me/local.html,Local,2025-04-25T12:00:00Z\n"
        "not a valid url,Invalid,2025-04-26T12:00:00Z\n",
        encoding="utf-8",
    )

    domain_only = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["domain"])
    web_only = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["web_history"])

    assert [unit.source_entity_type for unit in domain_only.units] == ["domain"]
    assert domain_only.units[0].metadata["domain"] == "example.com"
    assert domain_only.edges == []
    assert all(unit.source_entity_type == "web_history" for unit in web_only.units)
    assert web_only.edges == []


def test_browser_history_csv_respects_since_and_entity_type_filters(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "url,title,last_visit_time\n"
        "https://example.com/old,Old,2025-04-24T12:00:00Z\n"
        "https://example.com/new,New,2025-04-26T12:00:00Z\n",
        encoding="utf-8",
    )

    skipped = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["bookmark"])
    assert skipped.units == []
    assert skipped.edges == []

    result = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(
        since=SyncState(
            source_project="browser_history_csv",
            source_entity_type="web_history",
            last_sync_at=datetime(2025, 4, 25, tzinfo=timezone.utc),
        ),
        entity_types=["web_history"],
    )

    assert [unit.title for unit in result.units] == ["New"]


def test_browser_history_csv_missing_path_or_url_returns_empty_result(tmp_path):
    missing = BrowserHistoryCsvAdapter(path=str(tmp_path / "missing.csv")).ingest()
    assert missing.units == []
    assert missing.edges == []

    csv_path = tmp_path / "history.csv"
    csv_path.write_text("title,visit_count\nNo URL,4\n", encoding="utf-8")
    malformed = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest()
    assert malformed.units == []
    assert malformed.edges == []


def test_browser_history_csv_preserves_supported_referrer_columns(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "url,title,referrer_url,from_url,source_url\n"
        "https://example.com/one,One,https://search.example/?q=one,,\n"
        "https://example.com/two,Two,,https://example.com/one,\n"
        "https://example.com/three,Three,,,\n",
        encoding="utf-8",
    )

    result = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest()

    units = {unit.title: unit for unit in result.units}
    assert units["One"].metadata["referrer_url"] == "https://search.example/?q=one"
    assert units["Two"].metadata["referrer_url"] == "https://example.com/one"
    assert "referrer_url" not in units["Three"].metadata


def test_browser_history_csv_emits_visit_hour_aggregates_and_edges(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "url,title,visit_time,last_visit_time\n"
        "https://example.com/one,One,2025-04-24T12:15:00Z,2025-04-24T12:45:00Z\n"
        "https://example.com/two,Two,2025-04-24T13:05:00Z,\n"
        "https://example.com/unknown,Unknown,not a date,\n",
        encoding="utf-8",
    )

    hours_only = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["visit_hour"])
    combined = BrowserHistoryCsvAdapter(path=str(csv_path)).ingest(entity_types=["web_history", "visit_hour"])

    assert BrowserHistoryCsvAdapter().entity_types == ["web_history", "domain", "visit_hour"]
    hours = {unit.metadata["visit_hour"]: unit for unit in hours_only.units}
    assert sorted(hours) == [
        "2025-04-24T12:00:00+00:00",
        "2025-04-24T13:00:00+00:00",
    ]
    assert hours["2025-04-24T12:00:00+00:00"].metadata["visit_count"] == 1
    assert hours["2025-04-24T13:00:00+00:00"].metadata["visit_count"] == 1
    assert hours_only.edges == []

    hour_edges = [edge for edge in combined.edges if edge.metadata["relation_type"] == "visit_hour"]
    assert len(hour_edges) == 2
    assert {edge.relation for edge in hour_edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in hour_edges} == {EdgeSource.SOURCE}
