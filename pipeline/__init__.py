"""zot-research-engine pipeline package.

Each module implements one stage of the research-discovery pipeline:
fetching (openalex, arxiv, crossref), enrichment (semantic_scholar),
free offline relevance ranking (local_ranker), optional Gemini enrichment of
the shortlist (gemini_ranker), deduplication (deduplicate), storage
(zotero_client), and reporting (report).
"""
