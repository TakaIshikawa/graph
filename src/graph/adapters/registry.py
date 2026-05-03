"""Adapter registry."""

from __future__ import annotations

from graph.adapters.base import SourceAdapter
from graph.adapters.atom import AtomAdapter
from graph.adapters.bibdesk import BibDeskAdapter
from graph.adapters.bibtex import BibtexAdapter
from graph.adapters.bookmarks import BookmarksAdapter
from graph.adapters.bookmarks_html import BookmarksHtmlAdapter
from graph.adapters.browser_history_csv import BrowserHistoryCsvAdapter
from graph.adapters.chatgpt_json import ChatGptJsonAdapter
from graph.adapters.csv_adapter import CsvAdapter
from graph.adapters.csv_rows import CsvRowsAdapter
from graph.adapters.crossref import CrossrefAdapter
from graph.adapters.csl_json import CslJsonAdapter
from graph.adapters.daily_journal import DailyJournalAdapter
from graph.adapters.discord_json import DiscordJsonAdapter
from graph.adapters.email import EmailAdapter
from graph.adapters.enex import EnexAdapter
from graph.adapters.feed import FeedAdapter
from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.git_adapter import GitAdapter
from graph.adapters.google_keep import GoogleKeepAdapter
from graph.adapters.html import HtmlAdapter
from graph.adapters.hypothesis import HypothesisAdapter
from graph.adapters.ical import ICalAdapter
from graph.adapters.ipynb import IpynbAdapter
from graph.adapters.jats import JatsAdapter
from graph.adapters.jsonl_adapter import JsonlAdapter
from graph.adapters.jsonl_notes import JsonlNotesAdapter
from graph.adapters.kindle import KindleAdapter
from graph.adapters.logseq import LogseqAdapter
from graph.adapters.markdown import MarkdownAdapter
from graph.adapters.markdown_callouts import MarkdownCalloutsAdapter
from graph.adapters.markdown_definitions import MarkdownDefinitionsAdapter
from graph.adapters.markdown_links import MarkdownLinksAdapter
from graph.adapters.markdown_notes import MarkdownNotesAdapter
from graph.adapters.markdown_tasks import MarkdownTasksAdapter
from graph.adapters.mastodon import MastodonAdapter
from graph.adapters.max_adapter import MaxAdapter
from graph.adapters.me import MeAdapter
from graph.adapters.mediawiki import MediaWikiAdapter
from graph.adapters.notion_markdown import NotionMarkdownAdapter
from graph.adapters.opml import OpmlAdapter
from graph.adapters.obsidian_canvas import ObsidianCanvasAdapter
from graph.adapters.org import OrgAdapter
from graph.adapters.pdf import PdfAdapter
from graph.adapters.pinboard import PinboardAdapter
from graph.adapters.pocket import PocketAdapter
from graph.adapters.pocket_csv import PocketCsvAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.raindrop import RaindropAdapter
from graph.adapters.raindrop_csv import RaindropCsvAdapter
from graph.adapters.raindrop_json import RaindropJsonAdapter
from graph.adapters.readwise import ReadwiseAdapter
from graph.adapters.readwise_csv import ReadwiseCsvAdapter
from graph.adapters.roam import RoamAdapter
from graph.adapters.safari_bookmarks import SafariBookmarksAdapter
from graph.adapters.slack_json import SlackJsonAdapter
from graph.adapters.sota import SOTAAdapter
from graph.adapters.sqlite_query_log import SqliteQueryLogAdapter
from graph.adapters.ris import RisAdapter
from graph.adapters.text import TextAdapter
from graph.adapters.text_outline import TextOutlineAdapter
from graph.adapters.tana_paste import TanaPasteAdapter
from graph.adapters.transcript import TranscriptAdapter
from graph.adapters.twitter_archive import TwitterArchiveAdapter
from graph.adapters.webvtt import WebVttAdapter
from graph.adapters.yaml_adapter import YamlAdapter
from graph.adapters.zotero_rdf import ZoteroRdfAdapter

_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "atom": AtomAdapter,
    "forty_two": FortyTwoAdapter,
    "max": MaxAdapter,
    "presence": PresenceAdapter,
    "me": MeAdapter,
    "mediawiki": MediaWikiAdapter,
    "markdown": MarkdownAdapter,
    "markdown_callouts": MarkdownCalloutsAdapter,
    "markdown_definitions": MarkdownDefinitionsAdapter,
    "markdown_links": MarkdownLinksAdapter,
    "markdown_notes": MarkdownNotesAdapter,
    "markdown_tasks": MarkdownTasksAdapter,
    "mastodon": MastodonAdapter,
    "kindle": KindleAdapter,
    "sota": SOTAAdapter,
    "feed": FeedAdapter,
    "bookmarks": BookmarksAdapter,
    "bookmarks_html": BookmarksHtmlAdapter,
    "browser_history_csv": BrowserHistoryCsvAdapter,
    "chatgpt_json": ChatGptJsonAdapter,
    "discord_json": DiscordJsonAdapter,
    "csv": CsvAdapter,
    "csv_rows": CsvRowsAdapter,
    "jsonl": JsonlAdapter,
    "jsonl_notes": JsonlNotesAdapter,
    "daily_journal": DailyJournalAdapter,
    "yaml": YamlAdapter,
    "notion_markdown": NotionMarkdownAdapter,
    "opml": OpmlAdapter,
    "obsidian_canvas": ObsidianCanvasAdapter,
    "org": OrgAdapter,
    "pdf": PdfAdapter,
    "email": EmailAdapter,
    "enex": EnexAdapter,
    "text": TextAdapter,
    "text_outline": TextOutlineAdapter,
    "tana_paste": TanaPasteAdapter,
    "html": HtmlAdapter,
    "ical": ICalAdapter,
    "ipynb": IpynbAdapter,
    "bibdesk": BibDeskAdapter,
    "bibtex": BibtexAdapter,
    "csl_json": CslJsonAdapter,
    "crossref": CrossrefAdapter,
    "ris": RisAdapter,
    "jats": JatsAdapter,
    "git": GitAdapter,
    "google_keep": GoogleKeepAdapter,
    "transcript": TranscriptAdapter,
    "twitter_archive": TwitterArchiveAdapter,
    "webvtt": WebVttAdapter,
    "pocket": PocketAdapter,
    "pocket_csv": PocketCsvAdapter,
    "pinboard": PinboardAdapter,
    "raindrop": RaindropAdapter,
    "raindrop_csv": RaindropCsvAdapter,
    "raindrop_json": RaindropJsonAdapter,
    "safari_bookmarks": SafariBookmarksAdapter,
    "zotero_rdf": ZoteroRdfAdapter,
    "hypothesis": HypothesisAdapter,
    "readwise": ReadwiseAdapter,
    "readwise_csv": ReadwiseCsvAdapter,
    "roam": RoamAdapter,
    "logseq": LogseqAdapter,
    "sqlite_query_log": SqliteQueryLogAdapter,
    "slack_json": SlackJsonAdapter,
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
