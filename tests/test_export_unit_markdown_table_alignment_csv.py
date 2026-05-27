import csv
from io import StringIO

from graph.export import export_units_to_markdown_table_alignment_csv


def test_markdown_table_alignment_export_classifies_delimiter_cells():
    content = "\n".join(
        [
            "| Left | Right | Center | Default |",
            "| :--- | ---: | :---: | --- |",
            "| a | b | c | d |",
        ]
    )

    rows = list(csv.DictReader(StringIO(export_units_to_markdown_table_alignment_csv([{"id": "u1", "title": "Tables", "content": content}]))))

    assert rows == [
        {"unit_id": "u1", "title": "Tables", "table_start_line": "1", "column_index": "1", "alignment": "left", "delimiter_cell": ":---"},
        {"unit_id": "u1", "title": "Tables", "table_start_line": "1", "column_index": "2", "alignment": "right", "delimiter_cell": "---:"},
        {"unit_id": "u1", "title": "Tables", "table_start_line": "1", "column_index": "3", "alignment": "center", "delimiter_cell": ":---:"},
        {"unit_id": "u1", "title": "Tables", "table_start_line": "1", "column_index": "4", "alignment": "default", "delimiter_cell": "---"},
    ]


def test_markdown_table_alignment_export_ignores_malformed_delimiters_and_fences():
    content = "\n".join(
        [
            "| A | B |",
            "| --- | nope |",
            "```",
            "| Hidden | Table |",
            "| --- | --- |",
            "```",
            "A | B",
            "--- | ---",
        ]
    )

    rows = list(csv.DictReader(StringIO(export_units_to_markdown_table_alignment_csv([{"id": "u1", "content": content}]))))

    assert rows == []
