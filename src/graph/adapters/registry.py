"""Adapter registry."""

from __future__ import annotations

from graph.adapters.base import SourceAdapter
from graph.adapters.activitywatch_json import ActivityWatchJsonAdapter
from graph.adapters.amazon_orders_csv import AmazonOrdersCsvAdapter
from graph.adapters.audible_library_csv import AudibleLibraryCsvAdapter
from graph.adapters.asana_tasks_csv import AsanaTasksCsvAdapter
from graph.adapters.atom import AtomAdapter
from graph.adapters.bibdesk import BibDeskAdapter
from graph.adapters.bibtex import BibtexAdapter
from graph.adapters.bluesky_archive import BlueskyArchiveAdapter
from graph.adapters.boardgamegeek_collection_csv import BoardGameGeekCollectionCsvAdapter
from graph.adapters.bookmarks import BookmarksAdapter
from graph.adapters.bookmarks_html import BookmarksHtmlAdapter
from graph.adapters.browser_history_csv import BrowserHistoryCsvAdapter
from graph.adapters.calibre_sqlite import CalibreSqliteAdapter
from graph.adapters.calendar_events_csv import CalendarEventsCsvAdapter
from graph.adapters.chatgpt_json import ChatGptJsonAdapter
from graph.adapters.chrome_history import ChromeHistoryAdapter
from graph.adapters.chrome_reading_list_json import ChromeReadingListJsonAdapter
from graph.adapters.csv_adapter import CsvAdapter
from graph.adapters.csv_rows import CsvRowsAdapter
from graph.adapters.crossref import CrossrefAdapter
from graph.adapters.csl_json import CslJsonAdapter
from graph.adapters.daily_journal import DailyJournalAdapter
from graph.adapters.diigo import DiigoAdapter
from graph.adapters.discord_json import DiscordJsonAdapter
from graph.adapters.email import EmailAdapter
from graph.adapters.enex import EnexAdapter
from graph.adapters.feed import FeedAdapter
from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.git_adapter import GitAdapter
from graph.adapters.github_issues_json import GithubIssuesJsonAdapter
from graph.adapters.gitlab_issues_json import GitlabIssuesJsonAdapter
from graph.adapters.github_stars_csv import GithubStarsCsvAdapter
from graph.adapters.garmin_activities_csv import GarminActivitiesCsvAdapter
from graph.adapters.goodreads_library import GoodreadsLibraryAdapter
from graph.adapters.inaturalist_observations_csv import INaturalistObservationsCsvAdapter
from graph.adapters.instacart_orders_csv import InstacartOrdersCsvAdapter
from graph.adapters.google_calendar_json import GoogleCalendarJsonAdapter
from graph.adapters.google_calendar_takeout import GoogleCalendarTakeoutAdapter
from graph.adapters.google_photos_takeout import GooglePhotosTakeoutAdapter
from graph.adapters.google_keep import GoogleKeepAdapter
from graph.adapters.html import HtmlAdapter
from graph.adapters.hacker_news_saved import HackerNewsSavedAdapter
from graph.adapters.hypothesis import HypothesisAdapter
from graph.adapters.ical import ICalAdapter
from graph.adapters.instapaper import InstapaperAdapter
from graph.adapters.ipynb import IpynbAdapter
from graph.adapters.jats import JatsAdapter
from graph.adapters.jira_issues_csv import JiraIssuesCsvAdapter
from graph.adapters.linear_issues_json import LinearIssuesJsonAdapter
from graph.adapters.jsonl_adapter import JsonlAdapter
from graph.adapters.jsonl_notes import JsonlNotesAdapter
from graph.adapters.kindle import KindleAdapter
from graph.adapters.kindle_clippings import KindleClippingsAdapter
from graph.adapters.kobo_highlights_csv import KoboHighlightsCsvAdapter
from graph.adapters.logseq import LogseqAdapter
from graph.adapters.google_maps_timeline_json import GoogleMapsTimelineJsonAdapter
from graph.adapters.google_play_books_notes_csv import GooglePlayBooksNotesCsvAdapter
from graph.adapters.markdown import MarkdownAdapter
from graph.adapters.markdown_callouts import MarkdownCalloutsAdapter
from graph.adapters.markdown_definitions import MarkdownDefinitionsAdapter
from graph.adapters.markdown_frontmatter import MarkdownFrontmatterAdapter
from graph.adapters.markdown_links import MarkdownLinksAdapter
from graph.adapters.markdown_notes import MarkdownNotesAdapter
from graph.adapters.markdown_tasks import MarkdownTasksAdapter
from graph.adapters.mastodon import MastodonAdapter
from graph.adapters.matter import MatterAdapter
from graph.adapters.max_adapter import MaxAdapter
from graph.adapters.mbox import MboxAdapter
from graph.adapters.me import MeAdapter
from graph.adapters.mediawiki import MediaWikiAdapter
from graph.adapters.myanimelist_xml import MyAnimeListXmlAdapter
from graph.adapters.netflix_viewing_activity_csv import NetflixViewingActivityCsvAdapter
from graph.adapters.notion_markdown import NotionMarkdownAdapter
from graph.adapters.notion_export import NotionExportAdapter
from graph.adapters.evernote_export import EvernoteExportAdapter
from graph.adapters.bear_export import BearExportAdapter
from graph.adapters.apple_notes_export import AppleNotesExportAdapter
from graph.adapters.apple_health_workouts import AppleHealthWorkoutsAdapter
from graph.adapters.apple_music_library_csv import AppleMusicLibraryCsvAdapter
from graph.adapters.apple_reminders_csv import AppleRemindersCsvAdapter
from graph.adapters.day_one_json import DayOneJsonAdapter
from graph.adapters.firefox_places import FirefoxPlacesAdapter
from graph.adapters.things_csv import ThingsCsvAdapter
from graph.adapters.simplenote_export import SimplenoteExportAdapter
from graph.adapters.foam import FoamWorkspaceAdapter
from graph.adapters.tana import TanaAdapter
from graph.adapters.todoist import TodoistAdapter
from graph.adapters.are_na import AreNaAdapter
from graph.adapters.zotero_csv import ZoteroCsvAdapter
from graph.adapters.google_tasks import GoogleTasksAdapter
from graph.adapters.google_contacts_csv import GoogleContactsCsvAdapter
from graph.adapters.fitbit_sleep_csv import FitbitSleepCsvAdapter
from graph.adapters.airtable_csv import AirtableCsvAdapter
from graph.adapters.google_keep_export import GoogleKeepExportAdapter
from graph.adapters.archivebox_index_json import ArchiveBoxIndexJsonAdapter
from graph.adapters.opml import OpmlAdapter
from graph.adapters.obsidian_canvas import ObsidianCanvasAdapter
from graph.adapters.omnivore_json import OmnivoreJsonAdapter
from graph.adapters.org import OrgAdapter
from graph.adapters.pdf import PdfAdapter
from graph.adapters.pinboard import PinboardAdapter
from graph.adapters.pinboard_html_export import PinboardHtmlExportAdapter
from graph.adapters.peloton_workouts_csv import PelotonWorkoutsCsvAdapter
from graph.adapters.plain_text import PlainTextAdapter
from graph.adapters.pocket import PocketAdapter
from graph.adapters.pocket_csv import PocketCsvAdapter
from graph.adapters.pocket_export import PocketExportAdapter
from graph.adapters.podcasts_opml import PodcastsOpmlAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.raindrop import RaindropAdapter
from graph.adapters.raindrop_csv import RaindropCsvAdapter
from graph.adapters.raindrop_json import RaindropJsonAdapter
from graph.adapters.readwise import ReadwiseAdapter
from graph.adapters.readwise_csv import ReadwiseCsvAdapter
from graph.adapters.goodreads import GoodreadsAdapter
from graph.adapters.letterboxd import LetterboxdAdapter
from graph.adapters.rescuetime import RescueTimeAdapter
from graph.adapters.toggl import TogglAdapter
from graph.adapters.wakatime import WakaTimeAdapter
from graph.adapters.roam import RoamAdapter
from graph.adapters.safari_bookmarks import SafariBookmarksAdapter
from graph.adapters.safari_history import SafariHistoryAdapter
from graph.adapters.slack_json import SlackJsonAdapter
from graph.adapters.sleep_as_android_csv import SleepAsAndroidCsvAdapter
from graph.adapters.sota import SOTAAdapter
from graph.adapters.spotify_streaming_history import SpotifyStreamingHistoryAdapter
from graph.adapters.spotify_takeout import SpotifyTakeoutAdapter
from graph.adapters.stackoverflow_bookmarks_json import StackOverflowBookmarksJsonAdapter
from graph.adapters.strava_activities_json import StravaActivitiesJsonAdapter
from graph.adapters.storygraph_reading_history_csv import StoryGraphReadingHistoryCsvAdapter
from graph.adapters.steam_library_csv import SteamLibraryCsvAdapter
from graph.adapters.trello_board_json import TrelloBoardJsonAdapter
from graph.adapters.trakt_watch_history_csv import TraktWatchHistoryCsvAdapter
from graph.adapters.libby_loans_csv import LibbyLoansCsvAdapter
from graph.adapters.reddit_saved_csv import RedditSavedCsvAdapter
from graph.adapters.reddit_saved_json import RedditSavedJsonAdapter
from graph.adapters.sqlite_query_log import SqliteQueryLogAdapter
from graph.adapters.ris import RisAdapter
from graph.adapters.text import TextAdapter
from graph.adapters.text_outline import TextOutlineAdapter
from graph.adapters.tana_paste import TanaPasteAdapter
from graph.adapters.transcript import TranscriptAdapter
from graph.adapters.twitter_archive import TwitterArchiveAdapter
from graph.adapters.facebook_archive import FacebookArchiveAdapter
from graph.adapters.instagram_archive import InstagramArchiveAdapter
from graph.adapters.linkedin_archive import LinkedInArchiveAdapter
from graph.adapters.vcard import VCardAdapter
from graph.adapters.wallabag import WallabagAdapter
from graph.adapters.webvtt import WebVttAdapter
from graph.adapters.yaml_adapter import YamlAdapter
from graph.adapters.yaml_frontmatter import YamlFrontmatterAdapter
from graph.adapters.youtube_playlists_json import YouTubePlaylistsJsonAdapter
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
    "markdown_frontmatter": MarkdownFrontmatterAdapter,
    "markdown_links": MarkdownLinksAdapter,
    "markdown_notes": MarkdownNotesAdapter,
    "markdown_tasks": MarkdownTasksAdapter,
    "mastodon": MastodonAdapter,
    "kindle": KindleAdapter,
    "kindle_clippings": KindleClippingsAdapter,
    "sota": SOTAAdapter,
    "feed": FeedAdapter,
    "mbox": MboxAdapter,
    "bookmarks": BookmarksAdapter,
    "bookmarks_html": BookmarksHtmlAdapter,
    "browser_history_csv": BrowserHistoryCsvAdapter,
    "calibre_sqlite": CalibreSqliteAdapter,
    "calendar_events_csv": CalendarEventsCsvAdapter,
    "chrome_history": ChromeHistoryAdapter,
    "chrome_reading_list_json": ChromeReadingListJsonAdapter,
    "chatgpt_json": ChatGptJsonAdapter,
    "discord_json": DiscordJsonAdapter,
    "csv": CsvAdapter,
    "csv_rows": CsvRowsAdapter,
    "jsonl": JsonlAdapter,
    "jsonl_notes": JsonlNotesAdapter,
    "daily_journal": DailyJournalAdapter,
    "yaml": YamlAdapter,
    "notion_markdown": NotionMarkdownAdapter,
    "notion_export": NotionExportAdapter,
    "evernote_export": EvernoteExportAdapter,
    "bear_export": BearExportAdapter,
    "apple_notes_export": AppleNotesExportAdapter,
    "apple_health_workouts": AppleHealthWorkoutsAdapter,
    "apple_music_library_csv": AppleMusicLibraryCsvAdapter,
    "apple_reminders_csv": AppleRemindersCsvAdapter,
    "day_one_json": DayOneJsonAdapter,
    "things_csv": ThingsCsvAdapter,
    "firefox_places": FirefoxPlacesAdapter,
    "simplenote_export": SimplenoteExportAdapter,
    "google_keep_export": GoogleKeepExportAdapter,
    "opml": OpmlAdapter,
    "obsidian_canvas": ObsidianCanvasAdapter,
    "omnivore_json": OmnivoreJsonAdapter,
    "org": OrgAdapter,
    "pdf": PdfAdapter,
    "plain_text": PlainTextAdapter,
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
    "bluesky_archive": BlueskyArchiveAdapter,
    "csl_json": CslJsonAdapter,
    "crossref": CrossrefAdapter,
    "ris": RisAdapter,
    "jats": JatsAdapter,
    "git": GitAdapter,
    "google_keep": GoogleKeepAdapter,
    "transcript": TranscriptAdapter,
    "twitter_archive": TwitterArchiveAdapter,
    "facebook_archive": FacebookArchiveAdapter,
    "instagram_archive": InstagramArchiveAdapter,
    "linkedin_archive": LinkedInArchiveAdapter,
    "webvtt": WebVttAdapter,
    "pocket": PocketAdapter,
    "pocket_csv": PocketCsvAdapter,
    "pocket_export": PocketExportAdapter,
    "instapaper": InstapaperAdapter,
    "pinboard": PinboardAdapter,
    "pinboard_html_export": PinboardHtmlExportAdapter,
    "raindrop": RaindropAdapter,
    "raindrop_csv": RaindropCsvAdapter,
    "raindrop_json": RaindropJsonAdapter,
    "diigo": DiigoAdapter,
    "wallabag": WallabagAdapter,
    "safari_bookmarks": SafariBookmarksAdapter,
    "safari_history": SafariHistoryAdapter,
    "zotero_rdf": ZoteroRdfAdapter,
    "hypothesis": HypothesisAdapter,
    "readwise": ReadwiseAdapter,
    "readwise_csv": ReadwiseCsvAdapter,
    "matter": MatterAdapter,
    "goodreads": GoodreadsAdapter,
    "goodreads_library": GoodreadsLibraryAdapter,
    "letterboxd": LetterboxdAdapter,
    "rescuetime": RescueTimeAdapter,
    "toggl": TogglAdapter,
    "wakatime": WakaTimeAdapter,
    "roam": RoamAdapter,
    "logseq": LogseqAdapter,
    "sqlite_query_log": SqliteQueryLogAdapter,
    "slack_json": SlackJsonAdapter,
    "sleep_as_android_csv": SleepAsAndroidCsvAdapter,
    "vcard": VCardAdapter,
    "yaml_frontmatter": YamlFrontmatterAdapter,
    "foam_workspace": FoamWorkspaceAdapter,
    "tana": TanaAdapter,
    "todoist": TodoistAdapter,
    "are_na": AreNaAdapter,
    "zotero_csv": ZoteroCsvAdapter,
    "google_tasks": GoogleTasksAdapter,
    "google_contacts_csv": GoogleContactsCsvAdapter,
    "hacker_news_saved": HackerNewsSavedAdapter,
    "github_stars_csv": GithubStarsCsvAdapter,
    "airtable_csv": AirtableCsvAdapter,
    "google_calendar_json": GoogleCalendarJsonAdapter,
    "google_calendar_takeout": GoogleCalendarTakeoutAdapter,
    "google_photos_takeout": GooglePhotosTakeoutAdapter,
    "activitywatch_json": ActivityWatchJsonAdapter,
    "audible_library_csv": AudibleLibraryCsvAdapter,
    "fitbit_sleep_csv": FitbitSleepCsvAdapter,
    "spotify_streaming_history": SpotifyStreamingHistoryAdapter,
    "spotify_takeout": SpotifyTakeoutAdapter,
    "trakt_watch_history_csv": TraktWatchHistoryCsvAdapter,
    "netflix_viewing_activity_csv": NetflixViewingActivityCsvAdapter,
    "storygraph_reading_history_csv": StoryGraphReadingHistoryCsvAdapter,
    "myanimelist_xml": MyAnimeListXmlAdapter,
    "kobo_highlights_csv": KoboHighlightsCsvAdapter,
    "inaturalist_observations_csv": INaturalistObservationsCsvAdapter,
    "steam_library_csv": SteamLibraryCsvAdapter,
    "github_issues_json": GithubIssuesJsonAdapter,
    "gitlab_issues_json": GitlabIssuesJsonAdapter,
    "jira_issues_csv": JiraIssuesCsvAdapter,
    "trello_board_json": TrelloBoardJsonAdapter,
    "google_maps_timeline_json": GoogleMapsTimelineJsonAdapter,
    "google_play_books_notes_csv": GooglePlayBooksNotesCsvAdapter,
    "libby_loans_csv": LibbyLoansCsvAdapter,
    "boardgamegeek_collection_csv": BoardGameGeekCollectionCsvAdapter,
    "reddit_saved_csv": RedditSavedCsvAdapter,
    "reddit_saved_json": RedditSavedJsonAdapter,
    "archivebox_index_json": ArchiveBoxIndexJsonAdapter,
    "asana_tasks_csv": AsanaTasksCsvAdapter,
    "strava_activities_json": StravaActivitiesJsonAdapter,
    "garmin_activities_csv": GarminActivitiesCsvAdapter,
    "instacart_orders_csv": InstacartOrdersCsvAdapter,
    "amazon_orders_csv": AmazonOrdersCsvAdapter,
    "linear_issues_json": LinearIssuesJsonAdapter,
    "peloton_workouts_csv": PelotonWorkoutsCsvAdapter,
    "podcasts_opml": PodcastsOpmlAdapter,
    "stackoverflow_bookmarks_json": StackOverflowBookmarksJsonAdapter,
    "youtube_playlists_json": YouTubePlaylistsJsonAdapter,
}


def _normalize_name(name: str) -> str:
    """Normalize adapter name: strip whitespace, lowercase, replace hyphens with underscores."""
    return name.strip().lower().replace("-", "_")


def get_adapter(name: str, **kwargs: str) -> SourceAdapter:
    normalized = _normalize_name(name)
    cls = _ADAPTERS.get(normalized)
    if cls is None:
        available = sorted(_ADAPTERS.keys())
        raise KeyError(f"Unknown adapter: {normalized}. Available: {available}")
    return cls(**kwargs)


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS.keys())


def get_all_adapters(**kwargs: str) -> list[SourceAdapter]:
    return [cls(**kwargs) for cls in _ADAPTERS.values()]
