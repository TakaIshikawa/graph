from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export import export_units_to_abbreviation_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


@dataclass
class Unit:
    id: str
    content: str | None = None


def test_abbreviation_inventory_counts_definitions_and_acronym_tokens():
    result = rows(
        export_units_to_abbreviation_inventory_csv(
            [
                {
                    "id": "u1",
                    "content": "\n".join(
                        [
                            "*[api]: Application Programming Interface",
                            "*[CPU]: Central Processing Unit",
                            "The API calls the CPU, but GPU remains undefined.",
                        ]
                    ),
                }
            ]
        )
    )[0]

    assert result == {
        "unit_id": "u1",
        "abbreviation_definition_count": "2",
        "acronym_token_count": "3",
        "undefined_acronym_count": "1",
        "defined_acronyms": "API; CPU",
    }


def test_abbreviation_inventory_ignores_single_letters_common_words_and_fences():
    result = rows(
        export_units_to_abbreviation_inventory_csv(
            [
                {
                    "id": "u1",
                    "content": "\n".join(
                        [
                            "THE API AND CPU are mentioned.",
                            "A B C are not acronym tokens.",
                            "```md",
                            "*[API]: Ignored Definition",
                            "GPU CPU",
                            "```",
                            "*[API]: Application Programming Interface",
                        ]
                    ),
                }
            ]
        )
    )[0]

    assert result == {
        "unit_id": "u1",
        "abbreviation_definition_count": "1",
        "acronym_token_count": "2",
        "undefined_acronym_count": "1",
        "defined_acronyms": "API",
    }


def test_abbreviation_inventory_supports_object_units_and_path_write(tmp_path):
    output = tmp_path / "abbr.csv"

    result = export_units_to_abbreviation_inventory_csv(
        [Unit("o", "RAG uses LLM.\n*[rag]: Retrieval Augmented Generation")], output
    )

    assert result["path"] == str(output)
    assert result["unit_count"] == 1
    assert result["rows_exported"] == 1
    assert rows(output.read_text(encoding="utf-8"))[0] == {
        "unit_id": "o",
        "abbreviation_definition_count": "1",
        "acronym_token_count": "2",
        "undefined_acronym_count": "1",
        "defined_acronyms": "RAG",
    }
