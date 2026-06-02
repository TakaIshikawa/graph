from __future__ import annotations

from graph.adapters.google_forms_responses_csv import GoogleFormsResponsesCsvAdapter
from graph.adapters.registry import get_adapter


def test_google_forms_responses_csv_ingests_answers(tmp_path):
    export = tmp_path / "forms.csv"
    export.write_text("Timestamp,Email Address,Response ID,Favorite tool,Notes\n05/01/2026 10:30,ada@example.com,r1,Graph,Looks good\n", encoding="utf-8")

    unit = GoogleFormsResponsesCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "form_response"
    assert unit.metadata["respondent_email"] == "ada@example.com"
    assert unit.metadata["answers"] == {"Favorite tool": "Graph", "Notes": "Looks good"}
    assert "Favorite tool: Graph" in unit.content


def test_google_forms_responses_csv_skips_rows_without_answers(tmp_path):
    export = tmp_path / "forms.csv"
    export.write_text("Timestamp,Email Address,Question\n2026-05-01,ada@example.com,\n", encoding="utf-8")

    assert GoogleFormsResponsesCsvAdapter(path=str(export)).ingest().units == []


def test_google_forms_responses_csv_is_registered():
    assert isinstance(get_adapter("google-forms-responses-csv"), GoogleFormsResponsesCsvAdapter)
    assert isinstance(get_adapter("google_forms_responses_csv"), GoogleFormsResponsesCsvAdapter)
