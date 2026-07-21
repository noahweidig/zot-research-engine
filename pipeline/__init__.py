"""zot-research-engine pipeline package.

Each module implements one stage of the research-discovery pipeline:
fetching (openalex, arxiv, crossref), enrichment (semantic_scholar),
ranking (gemini_ranker), deduplication (deduplicate), storage
(zotero_client), and reporting (report).
"""
