"""Adapter registry."""

from __future__ import annotations

from graph.adapters.base import SourceAdapter
from graph.adapters.bibdesk import BibDeskAdapter
from graph.adapters.bibtex import BibtexAdapter
from graph.adapters.bookmarks import BookmarksAdapter
from graph.adapters.csv_adapter import CsvAdapter
from graph.adapters.crossref import CrossrefAdapter
from graph.adapters.csl_json import CslJsonAdapter
from graph.adapters.email import EmailAdapter
from graph.adapters.enex import EnexAdapter
from graph.adapters.feed import FeedAdapter
from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.git_adapter import GitAdapter
from graph.adapters.html import HtmlAdapter
from graph.adapters.hypothesis import HypothesisAdapter
from graph.adapters.ical import ICalAdapter
from graph.adapters.ipynb import IpynbAdapter
from graph.adapters.jats import JatsAdapter
from graph.adapters.jsonl_adapter import JsonlAdapter
from graph.adapters.kindle import KindleAdapter
from graph.adapters.markdown import MarkdownAdapter
from graph.adapters.max_adapter import MaxAdapter
from graph.adapters.me import MeAdapter
from graph.adapters.mediawiki import MediaWikiAdapter
from graph.adapters.opml import OpmlAdapter
from graph.adapters.org import OrgAdapter
from graph.adapters.pdf import PdfAdapter
from graph.adapters.pinboard import PinboardAdapter
from graph.adapters.pocket import PocketAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.raindrop import RaindropAdapter
from graph.adapters.readwise import ReadwiseAdapter
from graph.adapters.roam import RoamAdapter
from graph.adapters.sota import SOTAAdapter
from graph.adapters.ris import RisAdapter
from graph.adapters.text import TextAdapter
from graph.adapters.transcript import TranscriptAdapter
from graph.adapters.yaml_adapter import YamlAdapter
from graph.adapters.zotero_rdf import ZoteroRdfAdapter

_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "forty_two": FortyTwoAdapter,
    "max": MaxAdapter,
    "presence": PresenceAdapter,
    "me": MeAdapter,
    "mediawiki": MediaWikiAdapter,
    "markdown": MarkdownAdapter,
    "kindle": KindleAdapter,
    "sota": SOTAAdapter,
    "feed": FeedAdapter,
    "bookmarks": BookmarksAdapter,
    "csv": CsvAdapter,
    "jsonl": JsonlAdapter,
    "yaml": YamlAdapter,
    "opml": OpmlAdapter,
    "org": OrgAdapter,
    "pdf": PdfAdapter,
    "email": EmailAdapter,
    "enex": EnexAdapter,
    "text": TextAdapter,
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
    "transcript": TranscriptAdapter,
    "pocket": PocketAdapter,
    "pinboard": PinboardAdapter,
    "raindrop": RaindropAdapter,
    "zotero_rdf": ZoteroRdfAdapter,
    "hypothesis": HypothesisAdapter,
    "readwise": ReadwiseAdapter,
    "roam": RoamAdapter,
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
