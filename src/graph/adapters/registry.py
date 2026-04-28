"""Adapter registry."""

from __future__ import annotations

from graph.adapters.base import SourceAdapter
from graph.adapters.bookmarks import BookmarksAdapter
from graph.adapters.csv_adapter import CsvAdapter
from graph.adapters.email import EmailAdapter
from graph.adapters.feed import FeedAdapter
from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.html import HtmlAdapter
from graph.adapters.ical import ICalAdapter
from graph.adapters.ipynb import IpynbAdapter
from graph.adapters.jsonl_adapter import JsonlAdapter
from graph.adapters.kindle import KindleAdapter
from graph.adapters.markdown import MarkdownAdapter
from graph.adapters.max_adapter import MaxAdapter
from graph.adapters.me import MeAdapter
from graph.adapters.opml import OpmlAdapter
from graph.adapters.pdf import PdfAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.sota import SOTAAdapter
from graph.adapters.text import TextAdapter

_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "forty_two": FortyTwoAdapter,
    "max": MaxAdapter,
    "presence": PresenceAdapter,
    "me": MeAdapter,
    "markdown": MarkdownAdapter,
    "kindle": KindleAdapter,
    "sota": SOTAAdapter,
    "feed": FeedAdapter,
    "bookmarks": BookmarksAdapter,
    "csv": CsvAdapter,
    "jsonl": JsonlAdapter,
    "opml": OpmlAdapter,
    "pdf": PdfAdapter,
    "email": EmailAdapter,
    "text": TextAdapter,
    "html": HtmlAdapter,
    "ical": ICalAdapter,
    "ipynb": IpynbAdapter,
}


def get_adapter(name: str, **kwargs: str) -> SourceAdapter:
    cls = _ADAPTERS.get(name)
    if cls is None:
        raise KeyError(f"Unknown adapter: {name}. Available: {list(_ADAPTERS)}")
    return cls(**kwargs)


def list_adapters() -> list[str]:
    return list(_ADAPTERS)


def get_all_adapters(**kwargs: str) -> list[SourceAdapter]:
    return [cls(**kwargs) for cls in _ADAPTERS.values()]
