from enum import StrEnum


class SourceProject(StrEnum):
    FORTY_TWO = "forty_two"
    MAX = "max"
    PRESENCE = "presence"
    ME = "me"
    KINDLE = "kindle"
    SOTA = "sota"
    BOOKMARKS = "bookmarks"
    CSV = "csv"
    JSONL = "jsonl"


class ContentType(StrEnum):
    INSIGHT = "insight"
    FINDING = "finding"
    IDEA = "idea"
    DESIGN_BRIEF = "design_brief"
    ARTIFACT = "artifact"
    METADATA = "metadata"


class EdgeRelation(StrEnum):
    BUILDS_ON = "builds_on"
    CHALLENGES = "challenges"
    REFINES = "refines"
    DISCOVERS = "discovers"
    REPLICATES = "replicates"
    INSPIRES = "inspires"
    DERIVES_FROM = "derives_from"
    RELATES_TO = "relates_to"


class EdgeSource(StrEnum):
    SOURCE = "source"
    INFERRED = "inferred"
    MANUAL = "manual"
