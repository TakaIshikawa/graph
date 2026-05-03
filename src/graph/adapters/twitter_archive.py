"""Adapter for Twitter/X archive tweet exports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


HASHTAG_RE = re.compile(r"(?<![\w/])#([\w][\w-]*)", re.UNICODE)
MENTION_RE = re.compile(r"(?<![\w/])@([A-Za-z0-9_]{1,20})")
TWEET_URL_RE = re.compile(r"https?://(?:mobile\.)?(?:twitter|x)\.com/[^/\s]+/status/(\d+)", re.IGNORECASE)


class TwitterArchiveAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "twitter_archive"

    @property
    def entity_types(self) -> list[str]:
        return ["tweet"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        if entity_types and "tweet" not in entity_types:
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        sync_at = self._sync_datetime(since) if since else None
        references_by_source_id: dict[str, list[dict[str, str]]] = {}

        for path in self._archive_files(root):
            for index, tweet in enumerate(self._read_tweets(path)):
                unit = self._tweet_unit(tweet, path, root, index)
                if unit is None:
                    continue
                if sync_at and unit.updated_at <= sync_at:
                    continue
                result.units.append(unit)
                references_by_source_id[unit.source_id] = self._references(tweet)

        result.units.sort(key=lambda unit: (unit.created_at, unit.source_id))
        included_source_ids = {unit.source_id for unit in result.units}
        source_id_by_tweet_id = {
            self._tweet_index_key(unit.metadata.get("tweet_id")): unit.source_id
            for unit in result.units
            if self._tweet_index_key(unit.metadata.get("tweet_id"))
        }

        emitted_edges: set[tuple[str, str, str, str]] = set()
        for source_id in sorted(references_by_source_id):
            if source_id not in included_source_ids:
                continue
            for reference in references_by_source_id[source_id]:
                target_id = source_id_by_tweet_id.get(self._tweet_index_key(reference.get("tweet_id")), "")
                if not target_id or target_id == source_id or target_id not in included_source_ids:
                    continue
                relation_type = reference["relation_type"]
                tweet_id = reference["tweet_id"]
                edge_key = (source_id, target_id, relation_type, tweet_id)
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(source_id, target_id, relation_type, tweet_id),
                        from_unit_id=source_id,
                        to_unit_id=target_id,
                        relation=EdgeRelation.REFERENCES,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.TWITTER_ARCHIVE.value,
                            "from_entity_type": "tweet",
                            "to_entity_type": "tweet",
                            "relation_type": relation_type,
                            "referenced_tweet_id": tweet_id,
                        },
                    )
                )

        result.edges.sort(key=lambda edge: (edge.from_unit_id, edge.to_unit_id, edge.id))
        return result

    def _archive_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() in {".json", ".js"} else []
        if not root.is_dir():
            return []
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".js"} and self._looks_like_tweet_file(path)
        )

    def _looks_like_tweet_file(self, path: Path) -> bool:
        name = path.name.lower()
        return "tweet" in name or name in {"archive.js", "data.js"}

    def _read_tweets(self, path: Path) -> list[dict[str, Any]]:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            parsed = json.loads(self._json_payload(raw))
        except json.JSONDecodeError:
            return []
        return self._tweet_records(parsed)

    def _json_payload(self, raw: str) -> str:
        text = raw.strip()
        if text.startswith("{") or text.startswith("["):
            return text
        separator = text.find("=")
        if separator >= 0:
            text = text[separator + 1 :].strip()
        if text.endswith(";"):
            text = text[:-1].strip()
        return text

    def _tweet_records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [tweet for item in value for tweet in [self._unwrap_tweet(item)] if tweet is not None]
        if not isinstance(value, dict):
            return []
        for key in ("tweets", "Tweets", "items", "records"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [tweet for item in nested for tweet in [self._unwrap_tweet(item)] if tweet is not None]
        tweet = self._unwrap_tweet(value)
        return [tweet] if tweet is not None else []

    def _unwrap_tweet(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        tweet = value.get("tweet")
        if isinstance(tweet, dict):
            return tweet
        if self._tweet_id(value) or self._content(value):
            return value
        return None

    def _tweet_unit(
        self,
        tweet: dict[str, Any],
        path: Path,
        root: Path,
        index: int,
    ) -> KnowledgeUnit | None:
        content = self._content(tweet)
        if not content:
            return None

        tweet_id = self._tweet_id(tweet) or self._fallback_tweet_id(path, index, content)
        created_text = self._first(tweet, "created_at", "createdAt", "created", "date")
        created_at = self._parse_datetime(created_text) or datetime.now(timezone.utc)
        source_path = self._relative_path(path, root)
        entities = tweet.get("entities") if isinstance(tweet.get("entities"), dict) else {}
        extended_entities = tweet.get("extended_entities") if isinstance(tweet.get("extended_entities"), dict) else {}
        hashtags = self._hashtags(entities, content)
        mentions = self._mentions(entities, content)
        urls = self._urls(entities, content)
        media = self._media(entities) + self._media(extended_entities)
        references = self._references(tweet)
        screen_name = self._first(tweet, "screen_name", "user_screen_name", "username")

        metadata: dict[str, Any] = {
            "tweet_id": tweet_id,
            "created_at": created_text,
            "url": self._tweet_url(tweet, tweet_id, screen_name),
            "hashtags": hashtags,
            "mentions": mentions,
            "urls": urls,
            "media": media,
            "lang": self._first(tweet, "lang"),
            "source": self._first(tweet, "source"),
            "favorite_count": self._int_or_string(tweet.get("favorite_count")),
            "retweet_count": self._int_or_string(tweet.get("retweet_count")),
            "reply_to_tweet_id": self._first(tweet, "in_reply_to_status_id_str", "in_reply_to_status_id"),
            "reply_to_user_id": self._first(tweet, "in_reply_to_user_id_str", "in_reply_to_user_id"),
            "reply_to_screen_name": self._first(tweet, "in_reply_to_screen_name"),
            "quoted_tweet_id": self._first(tweet, "quoted_status_id_str", "quoted_status_id"),
            "retweeted": self._bool_or_none(tweet.get("retweeted")),
            "source_path": source_path,
            "path": source_path,
        }
        retweeted_id = self._retweeted_tweet_id(tweet)
        if retweeted_id:
            metadata["retweeted_tweet_id"] = retweeted_id
        if references:
            metadata["references"] = references

        tags = ["twitter"]
        tags.extend(f"hashtag-{self._tag_value(tag)}" for tag in hashtags)
        tags.extend(f"mention-{self._tag_value(mention['screen_name'])}" for mention in mentions if mention.get("screen_name"))

        return KnowledgeUnit(
            source_project=SourceProject.TWITTER_ARCHIVE,
            source_id=self._source_id(tweet_id),
            source_entity_type="tweet",
            title=self._title(content, created_at, screen_name),
            content=" ".join(content.split()),
            content_type=ContentType.ARTIFACT,
            metadata={key: value for key, value in metadata.items() if value != "" and value is not None},
            tags=self._dedupe(tags),
            created_at=created_at,
            updated_at=created_at,
        )

    def _references(self, tweet: dict[str, Any]) -> list[dict[str, str]]:
        references: list[dict[str, str]] = []
        reply_id = self._first(tweet, "in_reply_to_status_id_str", "in_reply_to_status_id")
        if reply_id:
            references.append({"relation_type": "twitter_reply", "tweet_id": reply_id})
        quote_id = self._first(tweet, "quoted_status_id_str", "quoted_status_id")
        if quote_id:
            references.append({"relation_type": "twitter_quote", "tweet_id": quote_id})
        for url in self._urls(tweet.get("entities") if isinstance(tweet.get("entities"), dict) else {}, ""):
            expanded = url.get("expanded_url") or url.get("url", "")
            match = TWEET_URL_RE.search(expanded)
            if match and not any(item["tweet_id"] == match.group(1) for item in references):
                references.append({"relation_type": "twitter_quote", "tweet_id": match.group(1)})
        return references

    def _hashtags(self, entities: Any, content: str) -> list[str]:
        tags: set[str] = set()
        if isinstance(entities, dict) and isinstance(entities.get("hashtags"), list):
            for item in entities["hashtags"]:
                if isinstance(item, dict):
                    tags.add(self._normalize_tag(self._first(item, "text", "tag")))
                elif isinstance(item, str):
                    tags.add(self._normalize_tag(item))
        for match in HASHTAG_RE.finditer(content):
            tags.add(self._normalize_tag(match.group(1)))
        return sorted(tag for tag in tags if tag)

    def _mentions(self, entities: Any, content: str) -> list[dict[str, str]]:
        mentions: dict[str, dict[str, str]] = {}
        if isinstance(entities, dict) and isinstance(entities.get("user_mentions"), list):
            for item in entities["user_mentions"]:
                if not isinstance(item, dict):
                    continue
                screen_name = self._first(item, "screen_name", "username")
                if not screen_name:
                    continue
                mentions[screen_name.lower()] = {
                    "screen_name": screen_name,
                    "name": self._first(item, "name"),
                    "id": self._first(item, "id_str", "id"),
                }
        for match in MENTION_RE.finditer(content):
            screen_name = match.group(1)
            mentions.setdefault(screen_name.lower(), {"screen_name": screen_name, "name": "", "id": ""})
        return sorted(mentions.values(), key=lambda item: item["screen_name"].lower())

    def _urls(self, entities: Any, content: str) -> list[dict[str, str]]:
        urls: list[dict[str, str]] = []
        if isinstance(entities, dict) and isinstance(entities.get("urls"), list):
            for item in entities["urls"]:
                if not isinstance(item, dict):
                    continue
                url = {
                    "url": self._first(item, "url"),
                    "expanded_url": self._first(item, "expanded_url", "expandedUrl"),
                    "display_url": self._first(item, "display_url", "displayUrl"),
                }
                if any(url.values()):
                    urls.append({key: value for key, value in url.items() if value})
        for match in TWEET_URL_RE.finditer(content):
            expanded_url = match.group(0)
            if not any(item.get("expanded_url") == expanded_url or item.get("url") == expanded_url for item in urls):
                urls.append({"url": expanded_url, "expanded_url": expanded_url})
        return urls

    def _media(self, entities: Any) -> list[dict[str, str]]:
        if not isinstance(entities, dict) or not isinstance(entities.get("media"), list):
            return []
        media: list[dict[str, str]] = []
        for item in entities["media"]:
            if not isinstance(item, dict):
                continue
            media_item = {
                "id": self._first(item, "id_str", "id"),
                "type": self._first(item, "type"),
                "url": self._first(item, "url"),
                "media_url": self._first(item, "media_url_https", "media_url"),
                "expanded_url": self._first(item, "expanded_url"),
                "display_url": self._first(item, "display_url"),
            }
            if any(media_item.values()):
                media.append({key: value for key, value in media_item.items() if value})
        return media

    def _source_id(self, tweet_id: str) -> str:
        return f"twitter_archive:{tweet_id}"

    def _edge_id(self, source_id: str, target_id: str, relation_type: str, tweet_id: str) -> str:
        raw = "|".join([SourceProject.TWITTER_ARCHIVE.value, relation_type, source_id, target_id, tweet_id])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"twitter-archive-references-{digest}"

    def _fallback_tweet_id(self, path: Path, index: int, content: str) -> str:
        raw = f"{path.as_posix()}:{index}:{content}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _tweet_id(self, tweet: dict[str, Any]) -> str:
        return self._first(tweet, "id_str", "id", "tweet_id", "tweetId")

    def _tweet_index_key(self, value: Any) -> str:
        return self._string(value)

    def _content(self, tweet: dict[str, Any]) -> str:
        return self._first(tweet, "full_text", "text", "content")

    def _title(self, content: str, created_at: datetime, screen_name: str) -> str:
        prefix = f"@{screen_name} " if screen_name else ""
        text = " ".join(content.split())
        if len(text) > 64:
            text = f"{text[:61].rstrip()}..."
        return f"{prefix}{created_at.date().isoformat()} {text}"

    def _tweet_url(self, tweet: dict[str, Any], tweet_id: str, screen_name: str) -> str:
        explicit = self._first(tweet, "tweet_url", "url", "permalink")
        if explicit:
            return explicit
        if screen_name:
            return f"https://twitter.com/{screen_name}/status/{tweet_id}"
        return f"https://twitter.com/i/web/status/{tweet_id}"

    def _retweeted_tweet_id(self, tweet: dict[str, Any]) -> str:
        retweeted_status = tweet.get("retweeted_status") or tweet.get("retweetedStatus")
        if isinstance(retweeted_status, dict):
            return self._tweet_id(retweeted_status)
        return self._first(tweet, "retweeted_status_id_str", "retweeted_status_id")

    def _tag_value(self, value: str) -> str:
        return "-".join(value.lower().split())

    def _normalize_tag(self, value: str) -> str:
        return value.strip().removeprefix("#").lower()

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return output

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = self._string(value)
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                try:
                    parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
                except ValueError:
                    return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _sync_datetime(self, since: SyncState) -> datetime:
        value = since.last_sync_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _relative_path(self, path: Path, root: Path) -> str:
        source_root = root.parent if root.is_file() else root
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _first(self, mapping: dict[str, Any], *keys: str) -> str:
        for key in keys:
            text = self._string(mapping.get(key))
            if text:
                return text
        return ""

    def _string(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _int_or_string(self, value: Any) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = self._string(value)
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text

    def _bool_or_none(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        return None
