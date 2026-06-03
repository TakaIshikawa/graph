"""Adapter registry."""

from __future__ import annotations

from graph.adapters.base import SourceAdapter
from graph.adapters.activitywatch_json import ActivityWatchJsonAdapter
from graph.adapters.acorns_activity_csv import AcornsActivityCsvAdapter
from graph.adapters.amazon_orders_csv import AmazonOrdersCsvAdapter
from graph.adapters.audible_library_csv import AudibleLibraryCsvAdapter
from graph.adapters.asana_projects_csv import AsanaProjectsCsvAdapter
from graph.adapters.asana_stories_json import AsanaStoriesJsonAdapter
from graph.adapters.asana_tasks_csv import AsanaTasksCsvAdapter
from graph.adapters.asana_tasks_json import AsanaTasksJsonAdapter
from graph.adapters.claude_conversations_json import ClaudeConversationsJsonAdapter
from graph.adapters.airtable_bases_csv import AirtableBasesCsvAdapter
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
from graph.adapters.calendly_events_csv import CalendlyEventsCsvAdapter
from graph.adapters.chatgpt_json import ChatGptJsonAdapter
from graph.adapters.chrome_history import ChromeHistoryAdapter
from graph.adapters.chrome_bookmarks_json import ChromeBookmarksJsonAdapter
from graph.adapters.chrome_reading_list_json import ChromeReadingListJsonAdapter
from graph.adapters.citi_credit_card_transactions_csv import CitiCreditCardTransactionsCsvAdapter
from graph.adapters.confluence_pages_json import ConfluencePagesJsonAdapter
from graph.adapters.csv_adapter import CsvAdapter
from graph.adapters.csv_rows import CsvRowsAdapter
from graph.adapters.crossref import CrossrefAdapter
from graph.adapters.csl_json import CslJsonAdapter
from graph.adapters.coursera_progress_csv import CourseraProgressCsvAdapter
from graph.adapters.daily_journal import DailyJournalAdapter
from graph.adapters.diigo import DiigoAdapter
from graph.adapters.discover_credit_card_transactions_csv import DiscoverCreditCardTransactionsCsvAdapter
from graph.adapters.discord_json import DiscordJsonAdapter
from graph.adapters.discord_saved_messages_json import DiscordSavedMessagesJsonAdapter
from graph.adapters.discord_threads_json import DiscordThreadsJsonAdapter
from graph.adapters.devto_bookmarks_json import DevtoBookmarksJsonAdapter
from graph.adapters.dropbox_paper_docs_json import DropboxPaperDocsJsonAdapter
from graph.adapters.email import EmailAdapter
from graph.adapters.enex import EnexAdapter
from graph.adapters.etrade_transactions_csv import EtradeTransactionsCsvAdapter
from graph.adapters.feed import FeedAdapter
from graph.adapters.forty_two import FortyTwoAdapter
from graph.adapters.git_adapter import GitAdapter
from graph.adapters.github_gists_json import GithubGistsJsonAdapter
from graph.adapters.github_commits_json import GithubCommitsJsonAdapter
from graph.adapters.github_discussions_json import GithubDiscussionsJsonAdapter
from graph.adapters.github_issues_json import GithubIssuesJsonAdapter
from graph.adapters.github_releases_json import GithubReleasesJsonAdapter
from graph.adapters.github_notifications_json import GithubNotificationsJsonAdapter
from graph.adapters.github_review_comments_json import GithubReviewCommentsJsonAdapter
from graph.adapters.gitlab_issues_json import GitlabIssuesJsonAdapter
from graph.adapters.gitlab_merge_requests_json import GitlabMergeRequestsJsonAdapter
from graph.adapters.gitlab_pipelines_json import GitlabPipelinesJsonAdapter
from graph.adapters.github_stars_csv import GithubStarsCsvAdapter
from graph.adapters.garmin_activities_csv import GarminActivitiesCsvAdapter
from graph.adapters.goodreads_library import GoodreadsLibraryAdapter
from graph.adapters.inaturalist_observations_csv import INaturalistObservationsCsvAdapter
from graph.adapters.instacart_orders_csv import InstacartOrdersCsvAdapter
from graph.adapters.google_calendar_json import GoogleCalendarJsonAdapter
from graph.adapters.google_calendar_takeout import GoogleCalendarTakeoutAdapter
from graph.adapters.google_docs_takeout_html import GoogleDocsTakeoutHtmlAdapter
from graph.adapters.google_fit_activity_csv import GoogleFitActivityCsvAdapter
from graph.adapters.google_photos_favorites_json import GooglePhotosFavoritesJsonAdapter
from graph.adapters.google_photos_takeout import GooglePhotosTakeoutAdapter
from graph.adapters.google_keep import GoogleKeepAdapter
from graph.adapters.google_location_semantic_history_json import GoogleLocationSemanticHistoryJsonAdapter
from graph.adapters.html import HtmlAdapter
from graph.adapters.hacker_news_saved import HackerNewsSavedAdapter
from graph.adapters.hacker_news_submissions_csv import HackerNewsSubmissionsCsvAdapter
from graph.adapters.hacker_news_upvoted_csv import HackerNewsUpvotedCsvAdapter
from graph.adapters.hypothesis import HypothesisAdapter
from graph.adapters.hypothesis_annotations_json import HypothesisAnnotationsJsonAdapter
from graph.adapters.ical import ICalAdapter
from graph.adapters.instapaper import InstapaperAdapter
from graph.adapters.ipynb import IpynbAdapter
from graph.adapters.jats import JatsAdapter
from graph.adapters.jira_issues_csv import JiraIssuesCsvAdapter
from graph.adapters.jira_projects_csv import JiraProjectsCsvAdapter
from graph.adapters.jira_worklogs_csv import JiraWorklogsCsvAdapter
from graph.adapters.linear_issues_json import LinearIssuesJsonAdapter
from graph.adapters.linear_issues_csv import LinearIssuesCsvAdapter
from graph.adapters.linear_comments_json import LinearCommentsJsonAdapter
from graph.adapters.linear_projects_json import LinearProjectsJsonAdapter
from graph.adapters.jsonl_adapter import JsonlAdapter
from graph.adapters.jsonl_notes import JsonlNotesAdapter
from graph.adapters.kindle import KindleAdapter
from graph.adapters.kindle_vocabulary_csv import KindleVocabularyCsvAdapter
from graph.adapters.kindle_clippings import KindleClippingsAdapter
from graph.adapters.kobo_highlights_csv import KoboHighlightsCsvAdapter
from graph.adapters.kobo_reading_stats_csv import KoboReadingStatsCsvAdapter
from graph.adapters.logseq import LogseqAdapter
from graph.adapters.m1_finance_activity_csv import M1FinanceActivityCsvAdapter
from graph.adapters.google_maps_timeline_json import GoogleMapsTimelineJsonAdapter
from graph.adapters.google_maps_reviews_json import GoogleMapsReviewsJsonAdapter
from graph.adapters.google_pay_transactions_csv import GooglePayTransactionsCsvAdapter
from graph.adapters.google_play_books_notes_csv import GooglePlayBooksNotesCsvAdapter
from graph.adapters.markdown import MarkdownAdapter
from graph.adapters.markdown_callouts import MarkdownCalloutsAdapter
from graph.adapters.markdown_definitions import MarkdownDefinitionsAdapter
from graph.adapters.markdown_frontmatter import MarkdownFrontmatterAdapter
from graph.adapters.markdown_links import MarkdownLinksAdapter
from graph.adapters.markdown_notes import MarkdownNotesAdapter
from graph.adapters.markdown_tasks import MarkdownTasksAdapter
from graph.adapters.mastodon import MastodonAdapter
from graph.adapters.mastodon_favourites_json import MastodonFavouritesJsonAdapter
from graph.adapters.mastodon_outbox_json import MastodonOutboxJsonAdapter
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
from graph.adapters.bear_notes_markdown import BearNotesMarkdownAdapter
from graph.adapters.apple_notes_export import AppleNotesExportAdapter
from graph.adapters.apple_health_workouts import AppleHealthWorkoutsAdapter
from graph.adapters.apple_calendar_events_csv import AppleCalendarEventsCsvAdapter
from graph.adapters.apple_books_library_csv import AppleBooksLibraryCsvAdapter
from graph.adapters.apple_music_library_csv import AppleMusicLibraryCsvAdapter
from graph.adapters.apple_podcasts_history_csv import ApplePodcastsHistoryCsvAdapter
from graph.adapters.medium_bookmarks_json import MediumBookmarksJsonAdapter
from graph.adapters.monarch_money_transactions_csv import MonarchMoneyTransactionsCsvAdapter
from graph.adapters.monzo_transactions_csv import MonzoTransactionsCsvAdapter
from graph.adapters.overcast_starred_episodes_json import OvercastStarredEpisodesJsonAdapter
from graph.adapters.paprika_recipes_json import PaprikaRecipesJsonAdapter
from graph.adapters.patreon_memberships_csv import PatreonMembershipsCsvAdapter
from graph.adapters.paypal_activity_csv import PaypalActivityCsvAdapter
from graph.adapters.personal_capital_transactions_csv import PersonalCapitalTransactionsCsvAdapter
from graph.adapters.revolut_transactions_csv import RevolutTransactionsCsvAdapter
from graph.adapters.rocket_money_transactions_csv import RocketMoneyTransactionsCsvAdapter
from graph.adapters.robinhood_account_activity_csv import RobinhoodAccountActivityCsvAdapter
from graph.adapters.robinhood_activity_csv import RobinhoodActivityCsvAdapter
from graph.adapters.runkeeper_activities_csv import RunkeeperActivitiesCsvAdapter
from graph.adapters.schwab_brokerage_transactions_csv import SchwabBrokerageTransactionsCsvAdapter
from graph.adapters.schwab_transactions_csv import SchwabTransactionsCsvAdapter
from graph.adapters.shopify_orders_csv import ShopifyOrdersCsvAdapter
from graph.adapters.simplifi_transactions_csv import SimplifiTransactionsCsvAdapter
from graph.adapters.wise_activity_csv import WiseActivityCsvAdapter
from graph.adapters.splitwise_expenses_csv import SplitwiseExpensesCsvAdapter
from graph.adapters.square_sales_csv import SquareSalesCsvAdapter
from graph.adapters.venmo_transactions_csv import VenmoTransactionsCsvAdapter
from graph.adapters.coinbase_transactions_csv import CoinbaseTransactionsCsvAdapter
from graph.adapters.stripe_balance_transactions_csv import StripeBalanceTransactionsCsvAdapter
from graph.adapters.quicken_transactions_csv import QuickenTransactionsCsvAdapter
from graph.adapters.ynab_transactions_csv import YnabTransactionsCsvAdapter
from graph.adapters.pocket_casts_listening_history_csv import PocketCastsListeningHistoryCsvAdapter
from graph.adapters.product_hunt_bookmarks_json import ProductHuntBookmarksJsonAdapter
from graph.adapters.product_hunt_votes_json import ProductHuntVotesJsonAdapter
from graph.adapters.apple_reminders_csv import AppleRemindersCsvAdapter
from graph.adapters.day_one_json import DayOneJsonAdapter
from graph.adapters.firefox_places import FirefoxPlacesAdapter
from graph.adapters.fitbit_daily_activity_csv import FitbitDailyActivityCsvAdapter
from graph.adapters.figma_files_json import FigmaFilesJsonAdapter
from graph.adapters.foursquare_checkins_csv import FoursquareCheckinsCsvAdapter
from graph.adapters.fidelity_activity_csv import FidelityActivityCsvAdapter
from graph.adapters.google_search_history_json import GoogleSearchHistoryJsonAdapter
from graph.adapters.things_csv import ThingsCsvAdapter
from graph.adapters.simplenote_export import SimplenoteExportAdapter
from graph.adapters.foam import FoamWorkspaceAdapter
from graph.adapters.tana import TanaAdapter
from graph.adapters.todoist import TodoistAdapter
from graph.adapters.todoist_tasks_csv import TodoistTasksCsvAdapter
from graph.adapters.todoist_tasks_json import TodoistTasksJsonAdapter
from graph.adapters.todoist_completed_csv import TodoistCompletedCsvAdapter
from graph.adapters.are_na import AreNaAdapter
from graph.adapters.zotero_csv import ZoteroCsvAdapter
from graph.adapters.google_tasks import GoogleTasksAdapter
from graph.adapters.google_tasks_json import GoogleTasksJsonAdapter
from graph.adapters.google_contacts_csv import GoogleContactsCsvAdapter
from graph.adapters.google_forms_responses_csv import GoogleFormsResponsesCsvAdapter
from graph.adapters.gmail_labels_csv import GmailLabelsCsvAdapter
from graph.adapters.google_scholar_library_csv import GoogleScholarLibraryCsvAdapter
from graph.adapters.fitbit_sleep_csv import FitbitSleepCsvAdapter
from graph.adapters.airtable_csv import AirtableCsvAdapter
from graph.adapters.google_keep_export import GoogleKeepExportAdapter
from graph.adapters.archivebox_index_json import ArchiveBoxIndexJsonAdapter
from graph.adapters.opml import OpmlAdapter
from graph.adapters.obsidian_canvas import ObsidianCanvasAdapter
from graph.adapters.obsidian_backlinks_markdown import ObsidianBacklinksMarkdownAdapter
from graph.adapters.omnivore_json import OmnivoreJsonAdapter
from graph.adapters.omnivore_highlights_json import OmnivoreHighlightsJsonAdapter
from graph.adapters.openai_conversations_json import OpenAIConversationsJsonAdapter
from graph.adapters.openlibrary_reading_log_csv import OpenLibraryReadingLogCsvAdapter
from graph.adapters.org import OrgAdapter
from graph.adapters.pdf import PdfAdapter
from graph.adapters.pinboard import PinboardAdapter
from graph.adapters.pinboard_html_export import PinboardHtmlExportAdapter
from graph.adapters.peloton_workouts_csv import PelotonWorkoutsCsvAdapter
from graph.adapters.plain_text import PlainTextAdapter
from graph.adapters.plaid_transactions_csv import PlaidTransactionsCsvAdapter
from graph.adapters.pocket import PocketAdapter
from graph.adapters.pocket_csv import PocketCsvAdapter
from graph.adapters.pocket_export import PocketExportAdapter
from graph.adapters.pocket_reading_list_csv import PocketReadingListCsvAdapter
from graph.adapters.pocket_articles_csv import PocketArticlesCsvAdapter
from graph.adapters.podcasts_opml import PodcastsOpmlAdapter
from graph.adapters.presence import PresenceAdapter
from graph.adapters.raindrop import RaindropAdapter
from graph.adapters.raindrop_bookmarks_csv import RaindropBookmarksCsvAdapter
from graph.adapters.raindrop_csv import RaindropCsvAdapter
from graph.adapters.raindrop_json import RaindropJsonAdapter
from graph.adapters.readwise import ReadwiseAdapter
from graph.adapters.readwise_csv import ReadwiseCsvAdapter
from graph.adapters.readwise_reader_documents_csv import ReadwiseReaderDocumentsCsvAdapter
from graph.adapters.goodreads import GoodreadsAdapter
from graph.adapters.letterboxd import LetterboxdAdapter
from graph.adapters.rescuetime import RescueTimeAdapter
from graph.adapters.rescuetime_daily_csv import RescueTimeDailyCsvAdapter
from graph.adapters.toggl import TogglAdapter
from graph.adapters.wakatime import WakaTimeAdapter
from graph.adapters.roam import RoamAdapter
from graph.adapters.rss_reader_starred_json import RssReaderStarredJsonAdapter
from graph.adapters.safari_bookmarks import SafariBookmarksAdapter
from graph.adapters.safari_history import SafariHistoryAdapter
from graph.adapters.safari_reading_list_csv import SafariReadingListCsvAdapter
from graph.adapters.slack_json import SlackJsonAdapter
from graph.adapters.slack_bookmarks_json import SlackBookmarksJsonAdapter
from graph.adapters.slack_threads_json import SlackThreadsJsonAdapter
from graph.adapters.sleep_as_android_csv import SleepAsAndroidCsvAdapter
from graph.adapters.sofi_activity_csv import SofiActivityCsvAdapter
from graph.adapters.sota import SOTAAdapter
from graph.adapters.spotify_saved_tracks_json import SpotifySavedTracksJsonAdapter
from graph.adapters.spotify_streaming_history import SpotifyStreamingHistoryAdapter
from graph.adapters.spotify_takeout import SpotifyTakeoutAdapter
from graph.adapters.stackoverflow_bookmarks_json import StackOverflowBookmarksJsonAdapter
from graph.adapters.stackoverflow_answers_json import StackOverflowAnswersJsonAdapter
from graph.adapters.strava_activities_json import StravaActivitiesJsonAdapter
from graph.adapters.storygraph_reading_history_csv import StoryGraphReadingHistoryCsvAdapter
from graph.adapters.storygraph_to_read_csv import StoryGraphToReadCsvAdapter
from graph.adapters.steam_library_csv import SteamLibraryCsvAdapter
from graph.adapters.tiller_transactions_csv import TillerTransactionsCsvAdapter
from graph.adapters.trello_board_json import TrelloBoardJsonAdapter
from graph.adapters.trello_cards_json import TrelloCardsJsonAdapter
from graph.adapters.trello_cards_csv import TrelloCardsCsvAdapter
from graph.adapters.trakt_watch_history_csv import TraktWatchHistoryCsvAdapter
from graph.adapters.tastytrade_activity_csv import TastytradeActivityCsvAdapter
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
from graph.adapters.us_bank_transactions_csv import UsBankTransactionsCsvAdapter
from graph.adapters.facebook_archive import FacebookArchiveAdapter
from graph.adapters.instagram_archive import InstagramArchiveAdapter
from graph.adapters.interactive_brokers_activity_csv import InteractiveBrokersActivityCsvAdapter
from graph.adapters.linkedin_archive import LinkedInArchiveAdapter
from graph.adapters.vcard import VCardAdapter
from graph.adapters.vanguard_activity_csv import VanguardActivityCsvAdapter
from graph.adapters.wallabag import WallabagAdapter
from graph.adapters.webull_brokerage_transactions_csv import WebullBrokerageTransactionsCsvAdapter
from graph.adapters.webvtt import WebVttAdapter
from graph.adapters.yaml_adapter import YamlAdapter
from graph.adapters.yaml_frontmatter import YamlFrontmatterAdapter
from graph.adapters.youtube_playlists_csv import YoutubePlaylistsCsvAdapter
from graph.adapters.youtube_comments_json import YoutubeCommentsJsonAdapter
from graph.adapters.youtube_playlists_json import YouTubePlaylistsJsonAdapter
from graph.adapters.zoom_recordings_csv import ZoomRecordingsCsvAdapter
from graph.adapters.zotero_library_csv import ZoteroLibraryCsvAdapter
from graph.adapters.clickup_tasks_csv import ClickUpTasksCsvAdapter
from graph.adapters.google_drive_files_csv import GoogleDriveFilesCsvAdapter
from graph.adapters.instapaper_highlights_csv import InstapaperHighlightsCsvAdapter
from graph.adapters.microsoft_edge_bookmarks_json import MicrosoftEdgeBookmarksJsonAdapter
from graph.adapters.miro_boards_csv import MiroBoardsCsvAdapter
from graph.adapters.notion_database_csv import NotionDatabaseCsvAdapter
from graph.adapters.notion_pages_csv import NotionPagesCsvAdapter
from graph.adapters.notion_comments_csv import NotionCommentsCsvAdapter
from graph.adapters.pocket_casts_queue_csv import PocketCastsQueueCsvAdapter
from graph.adapters.substack_posts_csv import SubstackPostsCsvAdapter
from graph.adapters.zotero_notes_markdown import ZoteroNotesMarkdownAdapter
from graph.adapters.zotero_rdf import ZoteroRdfAdapter
from graph.adapters.zotero_better_bibtex_json import ZoteroBetterBibtexJsonAdapter
from graph.adapters.workflowy_opml import WorkflowyOpmlAdapter
from graph.adapters.wikipedia_reading_list_csv import WikipediaReadingListCsvAdapter

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
    "mastodon_favourites_json": MastodonFavouritesJsonAdapter,
    "mastodon_outbox_json": MastodonOutboxJsonAdapter,
    "kindle": KindleAdapter,
    "kindle_clippings": KindleClippingsAdapter,
    "kindle_vocabulary_csv": KindleVocabularyCsvAdapter,
    "sota": SOTAAdapter,
    "feed": FeedAdapter,
    "mbox": MboxAdapter,
    "bookmarks": BookmarksAdapter,
    "bookmarks_html": BookmarksHtmlAdapter,
    "browser_history_csv": BrowserHistoryCsvAdapter,
    "calibre_sqlite": CalibreSqliteAdapter,
    "calendar_events_csv": CalendarEventsCsvAdapter,
    "calendly_events_csv": CalendlyEventsCsvAdapter,
    "chrome_history": ChromeHistoryAdapter,
    "chrome_bookmarks_json": ChromeBookmarksJsonAdapter,
    "chrome_reading_list_json": ChromeReadingListJsonAdapter,
    "citi_credit_card_transactions_csv": CitiCreditCardTransactionsCsvAdapter,
    "confluence_pages_json": ConfluencePagesJsonAdapter,
    "dropbox_paper_docs_json": DropboxPaperDocsJsonAdapter,
    "chatgpt_json": ChatGptJsonAdapter,
    "discover_credit_card_transactions_csv": DiscoverCreditCardTransactionsCsvAdapter,
    "discord_json": DiscordJsonAdapter,
    "discord_saved_messages_json": DiscordSavedMessagesJsonAdapter,
    "discord_threads_json": DiscordThreadsJsonAdapter,
    "csv": CsvAdapter,
    "csv_rows": CsvRowsAdapter,
    "jsonl": JsonlAdapter,
    "jsonl_notes": JsonlNotesAdapter,
    "daily_journal": DailyJournalAdapter,
    "coursera_progress_csv": CourseraProgressCsvAdapter,
    "yaml": YamlAdapter,
    "notion_markdown": NotionMarkdownAdapter,
    "notion_export": NotionExportAdapter,
    "evernote_export": EvernoteExportAdapter,
    "bear_export": BearExportAdapter,
    "bear_notes_markdown": BearNotesMarkdownAdapter,
    "apple_notes_export": AppleNotesExportAdapter,
    "apple_health_workouts": AppleHealthWorkoutsAdapter,
    "apple_calendar_events_csv": AppleCalendarEventsCsvAdapter,
    "apple_books_library_csv": AppleBooksLibraryCsvAdapter,
    "apple_music_library_csv": AppleMusicLibraryCsvAdapter,
    "apple_podcasts_history_csv": ApplePodcastsHistoryCsvAdapter,
    "medium_bookmarks_json": MediumBookmarksJsonAdapter,
    "overcast_starred_episodes_json": OvercastStarredEpisodesJsonAdapter,
    "paprika_recipes_json": PaprikaRecipesJsonAdapter,
    "pocket_casts_listening_history_csv": PocketCastsListeningHistoryCsvAdapter,
    "product_hunt_bookmarks_json": ProductHuntBookmarksJsonAdapter,
    "product_hunt_votes_json": ProductHuntVotesJsonAdapter,
    "apple_reminders_csv": AppleRemindersCsvAdapter,
    "day_one_json": DayOneJsonAdapter,
    "things_csv": ThingsCsvAdapter,
    "firefox_places": FirefoxPlacesAdapter,
    "foursquare_checkins_csv": FoursquareCheckinsCsvAdapter,
    "google_search_history_json": GoogleSearchHistoryJsonAdapter,
    "simplenote_export": SimplenoteExportAdapter,
    "google_keep_export": GoogleKeepExportAdapter,
    "opml": OpmlAdapter,
    "workflowy_opml": WorkflowyOpmlAdapter,
    "wikipedia_reading_list_csv": WikipediaReadingListCsvAdapter,
    "obsidian_canvas": ObsidianCanvasAdapter,
    "obsidian_backlinks_markdown": ObsidianBacklinksMarkdownAdapter,
    "omnivore_json": OmnivoreJsonAdapter,
    "omnivore_highlights_json": OmnivoreHighlightsJsonAdapter,
    "openai_conversations_json": OpenAIConversationsJsonAdapter,
    "openlibrary_reading_log_csv": OpenLibraryReadingLogCsvAdapter,
    "org": OrgAdapter,
    "pdf": PdfAdapter,
    "plain_text": PlainTextAdapter,
    "plaid_transactions_csv": PlaidTransactionsCsvAdapter,
    "email": EmailAdapter,
    "enex": EnexAdapter,
    "etrade_transactions_csv": EtradeTransactionsCsvAdapter,
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
    "us_bank_transactions_csv": UsBankTransactionsCsvAdapter,
    "facebook_archive": FacebookArchiveAdapter,
    "instagram_archive": InstagramArchiveAdapter,
    "interactive_brokers_activity_csv": InteractiveBrokersActivityCsvAdapter,
    "linkedin_archive": LinkedInArchiveAdapter,
    "webvtt": WebVttAdapter,
    "pocket": PocketAdapter,
    "pocket_csv": PocketCsvAdapter,
    "pocket_export": PocketExportAdapter,
    "pocket_reading_list_csv": PocketReadingListCsvAdapter,
    "pocket_articles_csv": PocketArticlesCsvAdapter,
    "instapaper": InstapaperAdapter,
    "pinboard": PinboardAdapter,
    "pinboard_html_export": PinboardHtmlExportAdapter,
    "raindrop": RaindropAdapter,
    "raindrop_bookmarks_csv": RaindropBookmarksCsvAdapter,
    "raindrop_csv": RaindropCsvAdapter,
    "raindrop_json": RaindropJsonAdapter,
    "diigo": DiigoAdapter,
    "wallabag": WallabagAdapter,
    "safari_bookmarks": SafariBookmarksAdapter,
    "safari_history": SafariHistoryAdapter,
    "safari_reading_list_csv": SafariReadingListCsvAdapter,
    "zotero_rdf": ZoteroRdfAdapter,
    "zotero_better_bibtex_json": ZoteroBetterBibtexJsonAdapter,
    "hypothesis": HypothesisAdapter,
    "hypothesis_annotations_json": HypothesisAnnotationsJsonAdapter,
    "readwise": ReadwiseAdapter,
    "readwise_csv": ReadwiseCsvAdapter,
    "readwise_reader_documents_csv": ReadwiseReaderDocumentsCsvAdapter,
    "matter": MatterAdapter,
    "goodreads": GoodreadsAdapter,
    "goodreads_library": GoodreadsLibraryAdapter,
    "letterboxd": LetterboxdAdapter,
    "rescuetime": RescueTimeAdapter,
    "toggl": TogglAdapter,
    "wakatime": WakaTimeAdapter,
    "roam": RoamAdapter,
    "logseq": LogseqAdapter,
    "m1_finance_activity_csv": M1FinanceActivityCsvAdapter,
    "sqlite_query_log": SqliteQueryLogAdapter,
    "slack_json": SlackJsonAdapter,
    "slack_bookmarks_json": SlackBookmarksJsonAdapter,
    "slack_threads_json": SlackThreadsJsonAdapter,
    "sleep_as_android_csv": SleepAsAndroidCsvAdapter,
    "sofi_activity_csv": SofiActivityCsvAdapter,
    "vanguard_activity_csv": VanguardActivityCsvAdapter,
    "vcard": VCardAdapter,
    "yaml_frontmatter": YamlFrontmatterAdapter,
    "foam_workspace": FoamWorkspaceAdapter,
    "tana": TanaAdapter,
    "todoist": TodoistAdapter,
    "todoist_tasks_csv": TodoistTasksCsvAdapter,
    "todoist_tasks_json": TodoistTasksJsonAdapter,
    "todoist_completed_csv": TodoistCompletedCsvAdapter,
    "are_na": AreNaAdapter,
    "zotero_csv": ZoteroCsvAdapter,
    "zotero_library_csv": ZoteroLibraryCsvAdapter,
    "google_tasks": GoogleTasksAdapter,
    "google_tasks_json": GoogleTasksJsonAdapter,
    "google_contacts_csv": GoogleContactsCsvAdapter,
    "google_forms_responses_csv": GoogleFormsResponsesCsvAdapter,
    "gmail_labels_csv": GmailLabelsCsvAdapter,
    "google_scholar_library_csv": GoogleScholarLibraryCsvAdapter,
    "hacker_news_saved": HackerNewsSavedAdapter,
    "hacker_news_submissions_csv": HackerNewsSubmissionsCsvAdapter,
    "hacker_news_upvoted_csv": HackerNewsUpvotedCsvAdapter,
    "github_stars_csv": GithubStarsCsvAdapter,
    "airtable_csv": AirtableCsvAdapter,
    "airtable_bases_csv": AirtableBasesCsvAdapter,
    "google_calendar_json": GoogleCalendarJsonAdapter,
    "google_calendar_takeout": GoogleCalendarTakeoutAdapter,
    "google_docs_takeout_html": GoogleDocsTakeoutHtmlAdapter,
    "google_photos_takeout": GooglePhotosTakeoutAdapter,
    "google_photos_favorites_json": GooglePhotosFavoritesJsonAdapter,
    "activitywatch_json": ActivityWatchJsonAdapter,
    "acorns_activity_csv": AcornsActivityCsvAdapter,
    "audible_library_csv": AudibleLibraryCsvAdapter,
    "fidelity_activity_csv": FidelityActivityCsvAdapter,
    "fitbit_daily_activity_csv": FitbitDailyActivityCsvAdapter,
    "google_fit_activity_csv": GoogleFitActivityCsvAdapter,
    "fitbit_sleep_csv": FitbitSleepCsvAdapter,
    "figma_files_json": FigmaFilesJsonAdapter,
    "spotify_saved_tracks_json": SpotifySavedTracksJsonAdapter,
    "spotify_streaming_history": SpotifyStreamingHistoryAdapter,
    "spotify_takeout": SpotifyTakeoutAdapter,
    "trakt_watch_history_csv": TraktWatchHistoryCsvAdapter,
    "tastytrade_activity_csv": TastytradeActivityCsvAdapter,
    "netflix_viewing_activity_csv": NetflixViewingActivityCsvAdapter,
    "storygraph_reading_history_csv": StoryGraphReadingHistoryCsvAdapter,
    "storygraph_to_read_csv": StoryGraphToReadCsvAdapter,
    "myanimelist_xml": MyAnimeListXmlAdapter,
    "kobo_highlights_csv": KoboHighlightsCsvAdapter,
    "kobo_reading_stats_csv": KoboReadingStatsCsvAdapter,
    "inaturalist_observations_csv": INaturalistObservationsCsvAdapter,
    "steam_library_csv": SteamLibraryCsvAdapter,
    "github_gists_json": GithubGistsJsonAdapter,
    "github_commits_json": GithubCommitsJsonAdapter,
    "github_discussions_json": GithubDiscussionsJsonAdapter,
    "github_issues_json": GithubIssuesJsonAdapter,
    "github_releases_json": GithubReleasesJsonAdapter,
    "github_notifications_json": GithubNotificationsJsonAdapter,
    "github_review_comments_json": GithubReviewCommentsJsonAdapter,
    "gitlab_issues_json": GitlabIssuesJsonAdapter,
    "gitlab_merge_requests_json": GitlabMergeRequestsJsonAdapter,
    "gitlab_pipelines_json": GitlabPipelinesJsonAdapter,
    "jira_issues_csv": JiraIssuesCsvAdapter,
    "jira_projects_csv": JiraProjectsCsvAdapter,
    "jira_worklogs_csv": JiraWorklogsCsvAdapter,
    "linear_comments_json": LinearCommentsJsonAdapter,
    "linear_projects_json": LinearProjectsJsonAdapter,
    "trello_board_json": TrelloBoardJsonAdapter,
    "trello_cards_json": TrelloCardsJsonAdapter,
    "trello_cards_csv": TrelloCardsCsvAdapter,
    "google_maps_timeline_json": GoogleMapsTimelineJsonAdapter,
    "google_maps_reviews_json": GoogleMapsReviewsJsonAdapter,
    "google_location_semantic_history_json": GoogleLocationSemanticHistoryJsonAdapter,
    "google_pay_transactions_csv": GooglePayTransactionsCsvAdapter,
    "google_play_books_notes_csv": GooglePlayBooksNotesCsvAdapter,
    "libby_loans_csv": LibbyLoansCsvAdapter,
    "monarch_money_transactions_csv": MonarchMoneyTransactionsCsvAdapter,
    "monzo_transactions_csv": MonzoTransactionsCsvAdapter,
    "paypal_activity_csv": PaypalActivityCsvAdapter,
    "patreon_memberships_csv": PatreonMembershipsCsvAdapter,
    "personal_capital_transactions_csv": PersonalCapitalTransactionsCsvAdapter,
    "revolut_transactions_csv": RevolutTransactionsCsvAdapter,
    "rocket_money_transactions_csv": RocketMoneyTransactionsCsvAdapter,
    "robinhood_account_activity_csv": RobinhoodAccountActivityCsvAdapter,
    "robinhood_activity_csv": RobinhoodActivityCsvAdapter,
    "runkeeper_activities_csv": RunkeeperActivitiesCsvAdapter,
    "schwab_brokerage_transactions_csv": SchwabBrokerageTransactionsCsvAdapter,
    "schwab_transactions_csv": SchwabTransactionsCsvAdapter,
    "shopify_orders_csv": ShopifyOrdersCsvAdapter,
    "simplifi_transactions_csv": SimplifiTransactionsCsvAdapter,
    "wise_activity_csv": WiseActivityCsvAdapter,
    "splitwise_expenses_csv": SplitwiseExpensesCsvAdapter,
    "square_sales_csv": SquareSalesCsvAdapter,
    "venmo_transactions_csv": VenmoTransactionsCsvAdapter,
    "coinbase_transactions_csv": CoinbaseTransactionsCsvAdapter,
    "stripe_balance_transactions_csv": StripeBalanceTransactionsCsvAdapter,
    "quicken_transactions_csv": QuickenTransactionsCsvAdapter,
    "ynab_transactions_csv": YnabTransactionsCsvAdapter,
    "boardgamegeek_collection_csv": BoardGameGeekCollectionCsvAdapter,
    "reddit_saved_csv": RedditSavedCsvAdapter,
    "reddit_saved_json": RedditSavedJsonAdapter,
    "rss_reader_starred_json": RssReaderStarredJsonAdapter,
    "archivebox_index_json": ArchiveBoxIndexJsonAdapter,
    "asana_tasks_csv": AsanaTasksCsvAdapter,
    "asana_tasks_json": AsanaTasksJsonAdapter,
    "claude_conversations_json": ClaudeConversationsJsonAdapter,
    "asana_projects_csv": AsanaProjectsCsvAdapter,
    "asana_stories_json": AsanaStoriesJsonAdapter,
    "strava_activities_json": StravaActivitiesJsonAdapter,
    "rescuetime_daily_csv": RescueTimeDailyCsvAdapter,
    "garmin_activities_csv": GarminActivitiesCsvAdapter,
    "instacart_orders_csv": InstacartOrdersCsvAdapter,
    "amazon_orders_csv": AmazonOrdersCsvAdapter,
    "tiller_transactions_csv": TillerTransactionsCsvAdapter,
    "webull_brokerage_transactions_csv": WebullBrokerageTransactionsCsvAdapter,
    "linear_issues_json": LinearIssuesJsonAdapter,
    "linear_issues_csv": LinearIssuesCsvAdapter,
    "peloton_workouts_csv": PelotonWorkoutsCsvAdapter,
    "podcasts_opml": PodcastsOpmlAdapter,
    "devto_bookmarks_json": DevtoBookmarksJsonAdapter,
    "stackoverflow_answers_json": StackOverflowAnswersJsonAdapter,
    "stackoverflow_bookmarks_json": StackOverflowBookmarksJsonAdapter,
    "youtube_playlists_json": YouTubePlaylistsJsonAdapter,
    "youtube_comments_json": YoutubeCommentsJsonAdapter,
    "youtube_playlists_csv": YoutubePlaylistsCsvAdapter,
    "zoom_recordings_csv": ZoomRecordingsCsvAdapter,
    "clickup_tasks_csv": ClickUpTasksCsvAdapter,
    "google_drive_files_csv": GoogleDriveFilesCsvAdapter,
    "instapaper_highlights_csv": InstapaperHighlightsCsvAdapter,
    "microsoft_edge_bookmarks_json": MicrosoftEdgeBookmarksJsonAdapter,
    "miro_boards_csv": MiroBoardsCsvAdapter,
    "notion_database_csv": NotionDatabaseCsvAdapter,
    "notion_pages_csv": NotionPagesCsvAdapter,
    "notion_comments_csv": NotionCommentsCsvAdapter,
    "pocket_casts_queue_csv": PocketCastsQueueCsvAdapter,
    "substack_posts_csv": SubstackPostsCsvAdapter,
    "zotero_notes_markdown": ZoteroNotesMarkdownAdapter,
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
