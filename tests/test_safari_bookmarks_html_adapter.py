from __future__ import annotations

from graph.adapters.safari_bookmarks_html import SafariBookmarksHtmlAdapter


def test_safari_bookmarks_html_extracts_links_and_nested_folders(tmp_path):
    export = tmp_path / "bookmarks.html"
    export.write_text(
        """
        <DL><p>
          <DT><H3 ADD_DATE="1">Work</H3>
          <DL><p>
            <DT><H3>Docs</H3>
            <DL><p>
              <DT><A HREF="https://example.test" ADD_DATE="1704067200" LAST_MODIFIED="1704153600" ICON="data:image/png;base64,x">Example</A>
            </DL><p>
          </DL><p>
        </DL><p>
        """,
        encoding="utf-8",
    )

    unit = SafariBookmarksHtmlAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "bookmark"
    assert unit.metadata["url"] == "https://example.test"
    assert unit.metadata["folder_path"] == ["Work", "Docs"]
    assert unit.metadata["add_date"] == "2024-01-01T00:00:00+00:00"
    assert unit.metadata["last_modified"] == "2024-01-02T00:00:00+00:00"
    assert unit.metadata["icon"] == "data:image/png;base64,x"


def test_safari_bookmarks_html_handles_missing_attributes(tmp_path):
    export = tmp_path / "bookmarks.html"
    export.write_text('<DL><p><DT><A HREF="https://minimal.test">Minimal</A></DL><p>', encoding="utf-8")

    unit = SafariBookmarksHtmlAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Minimal"
    assert unit.metadata["url"] == "https://minimal.test"
    assert "add_date" not in unit.metadata
