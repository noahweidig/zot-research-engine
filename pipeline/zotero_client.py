"""Zotero storage client.

Wraps ``pyzotero`` to ensure the inbox collection exists, expose existing DOIs
for deduplication, and batch-create ``journalArticle`` items for high-scoring
papers. Zotero failures are logged and swallowed so they never crash the run.
"""

from __future__ import annotations

import logging

from pyzotero import zotero

from config import PipelineConfig
from pipeline.models import RankedPaper

logger = logging.getLogger(__name__)

# Zotero's write API accepts at most 50 items per create call.
_ZOTERO_BATCH_SIZE = 50


class ZoteroClient:
    """Thin wrapper over pyzotero for the pipeline's needs."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the client and ensure the inbox collection exists.

        Args:
            config: The pipeline configuration.
        """
        self._config = config
        self._zot = zotero.Zotero(
            config.secrets.zotero_user_id,
            config.secrets.zotero_library_type,
            config.secrets.zotero_api_key,
        )
        self._inbox_name = config.zotero.inbox_collection
        self._inbox_key: str | None = None

    def _ensure_inbox(self) -> str | None:
        """Return the inbox collection key, creating the collection if needed.

        Returns:
            The collection key, or ``None`` if it could not be resolved.
        """
        if self._inbox_key:
            return self._inbox_key
        try:
            for coll in self._zot.collections():
                if coll["data"]["name"] == self._inbox_name:
                    self._inbox_key = coll["key"]
                    return self._inbox_key
            created = self._zot.create_collections([{"name": self._inbox_name}])
            self._inbox_key = created["successful"]["0"]["key"]
            logger.info("Created Zotero collection %r", self._inbox_name)
            return self._inbox_key
        except Exception as exc:  # noqa: BLE001 - Zotero failure is non-fatal
            logger.error("Could not ensure inbox collection: %s", exc)
            return None

    def existing_dois(self) -> set[str]:
        """Fetch all DOIs currently present in the library.

        Returns:
            A set of lower-cased DOIs (empty if the fetch fails).
        """
        dois: set[str] = set()
        try:
            items = self._zot.everything(self._zot.items())
        except Exception as exc:  # noqa: BLE001 - non-fatal
            logger.error("Could not fetch existing Zotero DOIs: %s", exc)
            return dois
        for item in items:
            doi = item.get("data", {}).get("DOI")
            if doi:
                dois.add(doi.strip().lower())
        logger.info("Found %d existing DOI(s) in Zotero library", len(dois))
        return dois

    def _build_item(self, paper: RankedPaper, inbox_key: str | None) -> dict:
        """Build a Zotero ``journalArticle`` item payload for a paper.

        Args:
            paper: The ranked paper.
            inbox_key: The inbox collection key to file the item under, if any.

        Returns:
            A Zotero item template dict.
        """
        creators = []
        for name in paper.authors:
            parts = name.rsplit(" ", 1)
            if len(parts) == 2:
                creators.append(
                    {
                        "creatorType": "author",
                        "firstName": parts[0],
                        "lastName": parts[1],
                    }
                )
            else:
                creators.append(
                    {"creatorType": "author", "lastName": name, "firstName": ""}
                )

        tags = [{"tag": "AI Suggested"}, {"tag": f"Score {paper.score}"}]
        tags += [{"tag": t} for t in paper.tags]

        return {
            "itemType": "journalArticle",
            "title": paper.title,
            "creators": creators,
            "abstractNote": paper.abstract or "",
            "publicationTitle": paper.journal or "",
            "date": str(paper.year) if paper.year else "",
            "DOI": paper.doi or "",
            "url": paper.url or "",
            "tags": tags,
            "collections": [inbox_key] if inbox_key else [],
        }

    def add_papers(self, papers: list[RankedPaper]) -> list[str]:
        """Batch-create Zotero items for the given papers and file them in inbox.

        Args:
            papers: The papers to add (already filtered by score by the caller).

        Returns:
            Titles of papers that were successfully added.
        """
        if not papers:
            return []

        inbox_key = self._ensure_inbox()
        added: list[str] = []

        for start in range(0, len(papers), _ZOTERO_BATCH_SIZE):
            batch = papers[start : start + _ZOTERO_BATCH_SIZE]
            templates = [self._build_item(p, inbox_key) for p in batch]
            try:
                resp = self._zot.create_items(templates)
            except Exception as exc:  # noqa: BLE001 - non-fatal
                logger.error("Zotero batch create failed: %s", exc)
                continue

            for idx_str in resp.get("successful", {}):
                title = batch[int(idx_str)].title
                added.append(title)
                logger.info("Added to Zotero: %s", title)

            for idx_str, err in resp.get("failed", {}).items():
                logger.error(
                    "Zotero rejected %r: %s", batch[int(idx_str)].title, err
                )

        logger.info("Added %d paper(s) to Zotero", len(added))
        return added
