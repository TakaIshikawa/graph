"""Adapter for WebVTT transcript files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graph.adapters.base import IngestResult, SourceAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit, SyncState


TIMING_RE = re.compile(
    r"^\s*(?P<start>\d{2}:(?:\d{2}:)?\d{2}\.\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:(?:\d{2}:)?\d{2}\.\d{3})(?:\s+.*)?$"
)
VOICE_RE = re.compile(r"^<v(?:\s+(?P<speaker>[^>]+))?>(?P<text>.*?)(?:</v>)?$", re.DOTALL)
SPEAKER_LABEL_RE = re.compile(r"^(?P<speaker>[A-Z][^:\n]{0,80}):\s+(?P<text>.+)$", re.DOTALL)


@dataclass(frozen=True)
class _Cue:
    start: str
    end: str
    text: str
    cue_index: int
    cue_id: str | None = None
    speaker: str | None = None


class WebVttAdapter(SourceAdapter):
    @property
    def name(self) -> str:
        return "webvtt"

    @property
    def entity_types(self) -> list[str]:
        return ["webvtt_transcript", "webvtt_cue"]

    def __init__(self, path: str = "", *, root_path: str = "") -> None:
        self.path = path or root_path

    def ingest(
        self,
        *,
        since: SyncState | None = None,
        entity_types: list[str] | None = None,
    ) -> IngestResult:
        result = IngestResult()
        requested_types = set(entity_types or self.entity_types)
        if requested_types.isdisjoint(self.entity_types):
            return result

        root = Path(self.path).expanduser()
        if not root.exists():
            return result

        source_root = root.parent if root.is_file() else root
        sync_at = self._sync_timestamp(since) if since else None
        for path in self._vtt_files(root):
            stat = path.stat()
            if sync_at is not None and stat.st_mtime <= sync_at:
                continue
            self._ingest_file(
                path,
                source_root,
                stat.st_size,
                stat.st_ctime,
                stat.st_mtime,
                requested_types,
                result,
            )

        return result

    def _vtt_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.lower() == ".vtt" else []
        if not root.is_dir():
            return []
        return sorted(path for path in root.rglob("*.vtt") if path.is_file())

    def _ingest_file(
        self,
        path: Path,
        source_root: Path,
        file_size: int,
        ctime: float,
        mtime: float,
        requested_types: set[str],
        result: IngestResult,
    ) -> None:
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return

        cues = self._parse(raw)
        if not cues:
            return

        source_path = self._relative_path(path, source_root)
        transcript_source_id = self._transcript_source_id(source_path)
        created_at = datetime.fromtimestamp(ctime, tz=timezone.utc)
        updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc)

        include_transcript = "webvtt_transcript" in requested_types
        include_cues = "webvtt_cue" in requested_types

        if include_transcript:
            result.units.append(
                KnowledgeUnit(
                    source_project=SourceProject.WEBVTT,
                    source_id=transcript_source_id,
                    source_entity_type="webvtt_transcript",
                    title=path.stem,
                    content="\n\n".join(cue.text for cue in cues),
                    content_type=ContentType.ARTIFACT,
                    metadata={
                        "path": source_path,
                        "source_path": source_path,
                        "file_size": file_size,
                        "cue_count": len(cues),
                        "first_timestamp": cues[0].start,
                        "last_timestamp": cues[-1].end,
                        "duration_range": f"{cues[0].start} --> {cues[-1].end}",
                    },
                    tags=["transcript", "webvtt"],
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )

        for cue in cues:
            cue_source_id = self._cue_source_id(source_path, cue)
            if include_cues:
                metadata = {
                    "path": source_path,
                    "source_path": source_path,
                    "start": cue.start,
                    "end": cue.end,
                    "cue_index": cue.cue_index,
                }
                if cue.cue_id:
                    metadata["cue_id"] = cue.cue_id
                if cue.speaker:
                    metadata["speaker"] = cue.speaker

                result.units.append(
                    KnowledgeUnit(
                        source_project=SourceProject.WEBVTT,
                        source_id=cue_source_id,
                        source_entity_type="webvtt_cue",
                        title=self._cue_title(path.stem, cue),
                        content=cue.text,
                        content_type=ContentType.INSIGHT,
                        metadata=metadata,
                        tags=["transcript", "webvtt"],
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                )

            if include_transcript and include_cues:
                result.edges.append(
                    KnowledgeEdge(
                        id=self._edge_id(transcript_source_id, cue_source_id),
                        from_unit_id=transcript_source_id,
                        to_unit_id=cue_source_id,
                        relation=EdgeRelation.CONTAINS,
                        source=EdgeSource.SOURCE,
                        metadata={
                            "source_project": SourceProject.WEBVTT.value,
                            "from_entity_type": "webvtt_transcript",
                            "to_entity_type": "webvtt_cue",
                            "relation_type": "webvtt_contains",
                            "source_path": source_path,
                            "cue_index": cue.cue_index,
                        },
                    )
                )

    def _parse(self, raw: str) -> list[_Cue]:
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        cues: list[_Cue] = []
        index = 0
        if lines and lines[0].lstrip("\ufeff").strip().startswith("WEBVTT"):
            index = 1

        pending_cue_id: str | None = None
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                pending_cue_id = None
                index += 1
                continue
            if line.startswith(("NOTE", "STYLE", "REGION")):
                pending_cue_id = None
                index = self._skip_block(lines, index + 1)
                continue

            if "-->" in line:
                timing_line = line
                index += 1
            elif index + 1 < len(lines) and "-->" in lines[index + 1]:
                pending_cue_id = line
                timing_line = lines[index + 1].strip()
                index += 2
            else:
                pending_cue_id = None
                index = self._skip_block(lines, index + 1)
                continue

            text_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                text_lines.append(lines[index].strip())
                index += 1

            cue = self._cue_from_lines(timing_line, text_lines, len(cues) + 1, pending_cue_id)
            if cue is not None:
                cues.append(cue)
            pending_cue_id = None

        return cues

    def _cue_from_lines(
        self,
        timing_line: str,
        text_lines: list[str],
        cue_index: int,
        cue_id: str | None,
    ) -> _Cue | None:
        match = TIMING_RE.match(timing_line)
        if match is None:
            return None

        text = "\n".join(line for line in text_lines if line).strip()
        if not text:
            return None

        speaker, text = self._extract_speaker(text)
        return _Cue(
            start=self._normalize_timestamp(match.group("start")),
            end=self._normalize_timestamp(match.group("end")),
            text=text,
            cue_index=cue_index,
            cue_id=cue_id,
            speaker=speaker,
        )

    def _extract_speaker(self, text: str) -> tuple[str | None, str]:
        voice_match = VOICE_RE.match(text)
        if voice_match is not None:
            speaker = (voice_match.group("speaker") or "").strip() or None
            return speaker, voice_match.group("text").strip()

        label_match = SPEAKER_LABEL_RE.match(text)
        if label_match is not None:
            return label_match.group("speaker").strip(), label_match.group("text").strip()

        return None, text

    def _normalize_timestamp(self, value: str) -> str:
        parts = value.split(":")
        if len(parts) == 2:
            return f"00:{parts[0]}:{parts[1]}"
        return value

    def _skip_block(self, lines: list[str], index: int) -> int:
        while index < len(lines) and lines[index].strip():
            index += 1
        return index

    def _cue_title(self, stem: str, cue: _Cue) -> str:
        speaker = f" {cue.speaker}" if cue.speaker else ""
        return f"{stem} {cue.start}{speaker}"

    def _transcript_source_id(self, source_path: str) -> str:
        return f"webvtt:{source_path}"

    def _cue_source_id(self, source_path: str, cue: _Cue) -> str:
        return f"webvtt:{source_path}:cue:{cue.cue_index}"

    def _edge_id(self, parent_source_id: str, child_source_id: str) -> str:
        raw = "|".join(
            [
                SourceProject.WEBVTT.value,
                EdgeRelation.CONTAINS.value,
                parent_source_id,
                child_source_id,
            ]
        )
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return f"webvtt-contains-{digest}"

    def _relative_path(self, path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _sync_timestamp(self, since: SyncState) -> float:
        if isinstance(since.last_sync_at, datetime):
            return since.last_sync_at.timestamp()
        return datetime.fromisoformat(str(since.last_sync_at)).timestamp()
