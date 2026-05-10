from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.pinboard_html_export import PinboardHtmlExportAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState

SAMPLE_HTML = """\
<!DOCTYPE NETSCAPE-Bookmark-file-1>
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Pinboard Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
<DT><A HREF="https://example.com/article" ADD_DATE="1704153600" PRIVATE="0" TOREAD="1" TAGS="python,testing">Example Article</A>
<DD>A useful article about testing
<DT><A HREF="https://example.com/tool" ADD_DATE="1704240000" PRIVATE="1" TOREAD="0" TAGS="devops,ci">CI Tool</A>
<DT><A HREF="https://example.com/plain" ADD_DATE="1704326400" PRIVATE="0" TOREAD="0" TAGS="">Plain Bookmark</A>
</DL><p>
"""


def test_parses_pinboard_html_export(tmp_path):
    f = tmp_path / "pinboard_export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    assert len(result.units) == 3

    urls = [u.metadata["url"] for u in result.units]
    assert "https://example.com/article" in urls
    assert "https://example.com/tool" in urls
    assert "https://example.com/plain" in urls


def test_tags_extracted_from_comma_separated(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    article = next(u for u in result.units if "article" in u.metadata["url"])
    assert article.tags == ["python", "testing"]

    tool = next(u for u in result.units if "tool" in u.metadata["url"])
    assert tool.tags == ["devops", "ci"]


def test_empty_tags_yields_empty_list(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    plain = next(u for u in result.units if "plain" in u.metadata["url"])
    assert plain.tags == []


def test_unix_timestamp_dates_parsed(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    article = next(u for u in result.units if "article" in u.metadata["url"])
    # 1704153600 = 2024-01-02 00:00:00 UTC
    assert article.created_at == datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    assert article.updated_at == article.created_at


def test_since_filtering(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    # Filter: only items after 2024-01-02 12:00:00 UTC
    since = SyncState(
        source_project="pinboard",
        source_entity_type="bookmark",
        last_sync_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
    )
    result = adapter.ingest(since=since)

    # article is at 2024-01-02 00:00:00 → excluded (<=)
    # tool is at 2024-01-03 00:00:00 → included
    # plain is at 2024-01-04 00:00:00 → included
    urls = [u.metadata["url"] for u in result.units]
    assert "https://example.com/article" not in urls
    assert "https://example.com/tool" in urls
    assert "https://example.com/plain" in urls


def test_private_and_toread_metadata(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    article = next(u for u in result.units if "article" in u.metadata["url"])
    assert article.metadata["private"] == "0"
    assert article.metadata["toread"] == "1"

    tool = next(u for u in result.units if "tool" in u.metadata["url"])
    assert tool.metadata["private"] == "1"
    assert tool.metadata["toread"] == "0"


def test_description_captured(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    article = next(u for u in result.units if "article" in u.metadata["url"])
    assert article.metadata["description"] == "A useful article about testing"
    assert "A useful article about testing" in article.content


def test_entity_types_filtering(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest(entity_types=["other_type"])
    assert len(result.units) == 0

    result = adapter.ingest(entity_types=["bookmark"])
    assert len(result.units) == 3


def test_source_project_and_content_type(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    for unit in result.units:
        assert unit.source_project == SourceProject.PINBOARD
        assert unit.source_entity_type == "bookmark"
        assert unit.content_type == ContentType.ARTIFACT


def test_empty_path_returns_empty():
    adapter = PinboardHtmlExportAdapter(path="")
    result = adapter.ingest()
    assert len(result.units) == 0


def test_missing_fields_handled(tmp_path):
    html = """\
<DL>
<DT><A HREF="https://example.com/minimal">Minimal</A>
</DL>
"""
    f = tmp_path / "minimal.html"
    f.write_text(html, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(f))
    result = adapter.ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Minimal"
    assert unit.metadata["url"] == "https://example.com/minimal"
    assert unit.tags == []


def test_adapter_name_and_entity_types():
    adapter = PinboardHtmlExportAdapter()
    assert adapter.name == "pinboard_html_export"
    assert adapter.entity_types == ["bookmark"]


def test_registered_in_registry():
    assert "pinboard_html_export" in list_adapters()
    adapter = get_adapter("pinboard_html_export", path="/tmp/test")
    assert isinstance(adapter, PinboardHtmlExportAdapter)


def test_directory_with_multiple_files(tmp_path):
    for i, name in enumerate(["a.html", "b.html"]):
        html = f'<DL><DT><A HREF="https://example.com/{i}" ADD_DATE="1704153600" TAGS="">Item {i}</A></DL>'
        (tmp_path / name).write_text(html, encoding="utf-8")

    adapter = PinboardHtmlExportAdapter(path=str(tmp_path))
    result = adapter.ingest()

    assert len(result.units) == 2


def test_idempotent_source_ids(tmp_path):
    f = tmp_path / "export.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")

    first = PinboardHtmlExportAdapter(path=str(f)).ingest()
    second = PinboardHtmlExportAdapter(path=str(f)).ingest()

    first_ids = [u.source_id for u in first.units]
    second_ids = [u.source_id for u in second.units]
    assert first_ids == second_ids
