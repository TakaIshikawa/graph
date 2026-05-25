"""Detect accessibility evidence requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_STANDARDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("WCAG", re.compile(r"\bwcag(?:\s*2(?:\.\d)?)?\b|web\s+content\s+accessibility\s+guidelines", re.I)),
    ("ADA", re.compile(r"\bada\b|americans\s+with\s+disabilities\s+act", re.I)),
    ("Section 508", re.compile(r"\bsection\s+508\b", re.I)),
    ("ARIA", re.compile(r"\baria\b|accessible\s+rich\s+internet\s+applications", re.I)),
)
_CUE_SPECS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("screen_reader", "visual", re.compile(r"\b(?:screen\s+reader|voiceover|nvda|jaws)\b", re.I)),
    ("keyboard_navigation", "motor", re.compile(r"\b(?:keyboard\s+(?:navigation|accessible|support)|tab\s+order|focus\s+(?:state|states|management)|focusable)\b", re.I)),
    ("captions", "audio", re.compile(r"\b(?:captions?|closed\s+captions?|subtitles?)\b", re.I)),
    ("transcript", "audio", re.compile(r"\btranscripts?\b", re.I)),
    ("alt_text", "visual", re.compile(r"\b(?:alt\s+text|alternative\s+text|image\s+description)\b", re.I)),
    ("contrast", "visual", re.compile(r"\b(?:contrast|color\s+contrast|contrast\s+ratio)\b", re.I)),
    ("colorblind", "visual", re.compile(r"\b(?:colorblind|color\s+blind|colourblind|colour\s+blind)\b", re.I)),
    ("reduced_motion", "cognitive", re.compile(r"\b(?:reduced\s+motion|motion\s+sensitivity|avoid\s+animations?|no\s+animations?)\b", re.I)),
    ("plain_language", "cognitive", re.compile(r"\b(?:plain\s+language|readability|cognitive\s+accessibility)\b", re.I)),
)
_MODALITY_ORDER = ("visual", "audio", "motor", "cognitive")


def detect_query_accessibility_requirement(query: str) -> dict[str, Any]:
    """Return accessibility cues, affected modalities, and retrieval recommendations."""
    normalized = _normalize_query(query)
    standards = _detected_standards(normalized)
    cues = _accessibility_cues(normalized)
    modalities = [modality for modality in _MODALITY_ORDER if any(cue["modality"] == modality for cue in cues)]
    requires = bool(standards or cues)
    return {
        "requires_accessibility_evidence": requires,
        "standards": standards,
        "accessibility_cues": cues,
        "affected_modalities": modalities,
        "recommendations": _recommendations(standards, modalities),
        "confidence": _confidence(standards, cues, modalities),
        "normalized_query": normalized,
    }


def _detected_standards(normalized_query: str) -> list[str]:
    return [name for name, pattern in _STANDARDS if pattern.search(normalized_query)]


def _accessibility_cues(normalized_query: str) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for kind, modality, pattern in _CUE_SPECS:
        for match in pattern.finditer(normalized_query):
            cues.append(
                {
                    "type": kind,
                    "cue": match.group(0).strip(),
                    "modality": modality,
                    "span": [match.start(), match.end()],
                }
            )
    cues.sort(key=lambda row: (row["span"][0], row["span"][1], row["type"]))
    return cues


def _recommendations(standards: list[str], modalities: list[str]) -> list[str]:
    recommendations = []
    if standards:
        recommendations.append("retrieve_named_accessibility_standards_and_compliance_evidence")
    if modalities:
        recommendations.append("prefer_sources_with_accessibility_testing_or_audit_details")
    if "visual" in modalities:
        recommendations.append("include_visual_accessibility_evidence_such_as_contrast_alt_text_or_screen_reader_support")
    if "audio" in modalities:
        recommendations.append("include_audio_accessibility_evidence_such_as_captions_or_transcripts")
    if "motor" in modalities:
        recommendations.append("include_keyboard_and_focus_navigation_evidence")
    if "cognitive" in modalities:
        recommendations.append("include_motion_readability_or_cognitive_accessibility_evidence")
    return recommendations


def _confidence(standards: list[str], cues: list[dict[str, Any]], modalities: list[str]) -> float:
    if standards and cues:
        return 0.95
    if standards:
        return 0.85
    if len(modalities) >= 2:
        return 0.8
    if cues:
        return 0.7
    return 0.0


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").casefold().split())
