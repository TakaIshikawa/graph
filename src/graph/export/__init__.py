"""Export helpers for graph reports."""

from graph.export.adjacency_markdown import export_graph_adjacency_markdown
from graph.export.adjacency_csv import export_graph_adjacency_csv
from graph.export.activity_heatmap_csv import export_units_to_activity_heatmap_csv
from graph.export.activitywatch_focus_sessions_csv import export_units_to_activitywatch_focus_sessions_csv
from graph.export.anki import export_units_to_anki_tsv
from graph.export.atom import export_units_to_atom
from graph.export.backlinks import export_unit_backlinks_markdown
from graph.export.bibliography_markdown import export_units_to_bibliography_markdown
from graph.export.bibtex import export_units_to_bibtex
from graph.export.bookmarks_html import export_units_to_bookmarks_html
from graph.export.concept_map import export_concept_map_markdown
from graph.export.context_pack import export_context_pack
from graph.export.csl_json import export_units_to_csl_json
from graph.export.chatgpt_digest_markdown import export_units_to_chatgpt_digest_markdown
from graph.export.collection_manifest_json import export_collection_manifest_json
from graph.export.collection_readme_markdown import export_collection_readme_markdown
from graph.export.collection_tag_index_markdown import export_collection_tag_index_markdown
from graph.export.cytoscape import export_graph_cytoscape
from graph.export.cytoscape_json import export_graph_cytoscape_json
from graph.export.cypher import export_graph_cypher
from graph.export.dot import export_graph_dot
from graph.export.dublin_core_xml import export_units_to_dublin_core_xml
from graph.export.duplicate_candidates import export_duplicate_candidates_markdown
from graph.export.csv_edges import export_graph_edges_csv
from graph.export.edge_csv import export_edges_to_csv
from graph.export.edge_adjacency_markdown import export_edge_adjacency_markdown
from graph.export.edge_coverage_csv import export_edge_coverage_csv
from graph.export.edge_type_matrix_csv import export_edge_type_matrix_csv
from graph.export.flashcards_markdown import export_units_to_flashcards_markdown
from graph.export.geojson import export_units_to_geojson
from graph.export.graphson import export_graphson
from graph.export.graphviz_dot import export_graphviz_dot
from graph.export.html_summary import export_graph_html_summary
from graph.export.graph_overview_html import render_graph_overview_html
from graph.export.html_report import render_search_html_report
from graph.export.hugo_bundle import export_units_to_hugo_bundle
from graph.export.ical import DATE_METADATA_KEYS, export_units_to_ics, unit_event_datetime
from graph.export.ical_timeline import export_units_to_ical_timeline
from graph.export.json import export_units_to_json
from graph.export.json_ld import export_units_to_json_ld
from graph.export.json_summary import export_graph_json_summary
from graph.export.kindle_review_queue_markdown import export_units_to_kindle_review_queue_markdown
from graph.export.llms_txt import export_units_to_llms_txt
from graph.export.markdown_timeline import export_units_to_markdown_timeline
from graph.export.metadata_key_matrix_csv import export_metadata_key_matrix_csv
from graph.export.metadata_completeness import export_metadata_completeness_markdown
from graph.export.mermaid_mindmap import export_units_to_mermaid_mindmap
from graph.export.ndjson import export_graph_ndjson, export_units_to_ndjson
from graph.export.neo4j_bulk_csv import export_graph_neo4j_bulk_csv
from graph.export.node_edge_csv import export_graph_node_edge_csv
from graph.export.opml import export_units_to_opml
from graph.export.org import export_units_to_org
from graph.export.orphan_units import export_orphan_units_markdown
from graph.export.relation_evidence import export_relation_evidence_markdown
from graph.export.rdf_turtle import export_graph_rdf_turtle
from graph.export.ris import export_units_to_ris
from graph.export.rss import export_units_to_rss
from graph.export.schema_inventory import export_unit_schema_inventory
from graph.export.source_coverage import export_source_coverage_markdown
from graph.export.source_entity_type_heatmap_csv import export_source_entity_type_heatmap_csv
from graph.export.source_inventory import export_source_inventory_csv
from graph.export.source_metadata_outliers_markdown import export_source_metadata_outliers_markdown
from graph.export.source_quality_markdown import export_source_quality_markdown
from graph.export.source_tag_summary_markdown import export_source_tag_summary_markdown
from graph.export.source_timeline_csv import export_source_timeline_csv
from graph.export.slack_participation_csv import export_units_to_slack_participation_csv
from graph.export.sqlite_snapshot import export_graph_sqlite
from graph.export.tag_activity_markdown import export_tag_activity_markdown
from graph.export.tag_cooccurrence_csv import export_tag_cooccurrence_csv
from graph.export.tag_glossary import export_tag_glossary_markdown
from graph.export.tag_source_matrix_csv import export_tag_source_matrix_csv
from graph.export.tag_timeline_csv import export_tag_timeline_csv
from graph.export.task_board_markdown import export_task_board_markdown
from graph.export.timelinejs import export_units_to_timelinejs
from graph.export.tiddlywiki_json import export_units_to_tiddlywiki_json
from graph.export.unit_csv import export_units_to_csv
from graph.export.unit_markdown_table import export_units_to_markdown_table
from graph.export.unit_yaml import export_units_to_yaml
from graph.export.units_jsonl import export_units_to_jsonl
from graph.export.vegalite_timeline import export_units_to_vegalite_timeline
from graph.export.zettelkasten_index import export_zettelkasten_index_markdown

try:
    from graph.export.gexf import export_graph_gexf
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise

    def export_graph_gexf(*args, **kwargs):
        from graph.export.gexf import export_graph_gexf as _export_graph_gexf

        return _export_graph_gexf(*args, **kwargs)


try:
    from graph.export.graphml import export_graph_graphml
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise

    def export_graph_graphml(*args, **kwargs):
        from graph.export.graphml import export_graph_graphml as _export_graph_graphml

        return _export_graph_graphml(*args, **kwargs)

__all__ = [
    "DATE_METADATA_KEYS",
    "export_units_to_bibliography_markdown",
    "export_units_to_bibtex",
    "export_units_to_bookmarks_html",
    "export_concept_map_markdown",
    "export_context_pack",
    "export_collection_manifest_json",
    "export_collection_readme_markdown",
    "export_collection_tag_index_markdown",
    "export_units_to_csl_json",
    "export_graph_cytoscape",
    "export_graph_cytoscape_json",
    "export_units_to_dublin_core_xml",
    "export_duplicate_candidates_markdown",
    "export_edge_adjacency_markdown",
    "export_edge_coverage_csv",
    "export_edge_type_matrix_csv",
    "export_edges_to_csv",
    "export_graph_edges_csv",
    "export_graph_adjacency_csv",
    "export_graph_adjacency_markdown",
    "export_units_to_activity_heatmap_csv",
    "export_units_to_activitywatch_focus_sessions_csv",
    "export_graph_cypher",
    "export_units_to_chatgpt_digest_markdown",
    "export_units_to_flashcards_markdown",
    "export_graph_dot",
    "export_graph_gexf",
    "export_units_to_geojson",
    "export_graph_graphml",
    "export_graph_json_summary",
    "export_graph_rdf_turtle",
    "export_graphson",
    "export_graphviz_dot",
    "export_graph_html_summary",
    "export_graph_sqlite",
    "export_units_to_hugo_bundle",
    "export_units_to_kindle_review_queue_markdown",
    "export_metadata_completeness_markdown",
    "export_metadata_key_matrix_csv",
    "export_graph_ndjson",
    "export_graph_neo4j_bulk_csv",
    "export_graph_node_edge_csv",
    "export_orphan_units_markdown",
    "export_relation_evidence_markdown",
    "export_units_to_ris",
    "export_units_to_rss",
    "export_source_coverage_markdown",
    "export_source_entity_type_heatmap_csv",
    "export_source_inventory_csv",
    "export_source_metadata_outliers_markdown",
    "export_source_quality_markdown",
    "export_source_tag_summary_markdown",
    "export_source_timeline_csv",
    "export_units_to_slack_participation_csv",
    "export_units_to_timelinejs",
    "export_units_to_tiddlywiki_json",
    "render_search_html_report",
    "render_graph_overview_html",
    "export_tag_activity_markdown",
    "export_tag_cooccurrence_csv",
    "export_tag_glossary_markdown",
    "export_tag_source_matrix_csv",
    "export_tag_timeline_csv",
    "export_task_board_markdown",
    "export_unit_backlinks_markdown",
    "export_unit_schema_inventory",
    "export_units_to_anki_tsv",
    "export_units_to_atom",
    "export_units_to_csv",
    "export_units_to_json",
    "export_units_to_json_ld",
    "export_units_to_jsonl",
    "export_units_to_llms_txt",
    "export_units_to_markdown_timeline",
    "export_units_to_markdown_table",
    "export_units_to_mermaid_mindmap",
    "export_units_to_ndjson",
    "export_units_to_opml",
    "export_units_to_org",
    "export_units_to_yaml",
    "export_units_to_ics",
    "export_units_to_ical_timeline",
    "export_units_to_vegalite_timeline",
    "export_zettelkasten_index_markdown",
    "unit_event_datetime",
]
