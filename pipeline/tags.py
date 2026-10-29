"""Tag normalization for Zotero.

Keeps tags lowercase, short and free of pipeline bookkeeping noise.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MAX_TAGS = 6
MAX_WORDS = 5

_DROPPED = frozenset({"ai suggested", "n/a", "todo"})
_SCORE_RE = re.compile(r"score\s*\d+")


def normalize_tags(tags: Iterable[str], max_tags: int = MAX_TAGS) -> list[str]:
    """Clean a list of raw tags.

    Lowercases, collapses whitespace, de-duplicates (order preserved), and drops
    ``ai suggested``, ``score N``, ``n/a``, ``todo``, tags ending in ``...`` and
    tags longer than 5 words. Caps the result at ``max_tags``.

    Args:
        tags: Raw tag strings.
        max_tags: Maximum number of tags to return.

    Returns:
        The cleaned tags.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = " ".join(str(raw).lower().split())
        if not tag or tag in seen:
            continue
        if tag in _DROPPED or _SCORE_RE.fullmatch(tag):
            continue
        if tag.endswith("...") or tag.endswith("…"):
            continue
        if len(tag.split()) > MAX_WORDS:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= max_tags:
            break
    return out
