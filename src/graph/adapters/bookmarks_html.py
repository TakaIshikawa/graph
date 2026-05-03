"""Adapter for browser bookmarks exported as Netscape HTML."""

from __future__ import annotations

from graph.adapters.bookmarks import BookmarksAdapter
from graph.types.enums import SourceProject


class BookmarksHtmlAdapter(BookmarksAdapter):
    source_project = SourceProject.BOOKMARKS_HTML

    @property
    def name(self) -> str:
        return "bookmarks_html"
