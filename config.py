"""Configuration loading and validation for zot-research-engine.

Loads secrets from environment variables (via a local ``.env`` file during
development, or GitHub Secrets in CI) and non-secret behavior from
``config.yaml``. Exposes a single typed :class:`PipelineConfig` dataclass that
the rest of the application imports.

Secrets are only ever passed to the libraries that need them — they are never
logged or written to disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Required environment variables. The pipeline refuses to start without these.
_REQUIRED_ENV_VARS: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "ZOTERO_API_KEY",
    "ZOTERO_USER_ID",
    "ZOTERO_LIBRARY_TYPE",
    "OPENALEX_EMAIL",
)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class SearchConfig:
    """Search behavior loaded from the ``search`` block of config.yaml."""

    queries: list[str]
    days_back: int
    max_results_per_query: int


@dataclass(frozen=True)
class RelevanceConfig:
    """Relevance scoring settings from the ``relevance`` block."""

    min_score: int
    interests: str


@dataclass(frozen=True)
class ZoteroConfig:
    """Zotero destination settings from the ``zotero`` block."""

    library_type: str
    inbox_collection: str


@dataclass(frozen=True)
class SourcesConfig:
    """Toggles for optional data sources from the ``sources`` block."""

    enable_semantic_scholar: bool
    enable_arxiv: bool
    enable_crossref: bool


@dataclass(frozen=True)
class PipelineSettings:
    """Runtime settings from the ``pipeline`` block."""

    log_to_file: bool
    log_file: str
    gemini_max_concurrency: int


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini model settings from the ``gemini`` block."""

    model: str
    temperature: float
    enrich: bool


@dataclass(frozen=True)
class RankerConfig:
    """Local (free) relevance-ranker settings from the ``ranker`` block.

    Attributes:
        thresholds: Four descending coverage cutoffs ``[t5, t4, t3, t2]`` that
            map IDF-weighted interest coverage onto the 1-5 score scale.
    """

    thresholds: list[float]


@dataclass(frozen=True)
class Secrets:
    """Secret values sourced exclusively from environment variables."""

    gemini_api_key: str = field(repr=False)
    zotero_api_key: str = field(repr=False)
    zotero_user_id: str
    zotero_library_type: str
    openalex_email: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return a redacted representation so secrets never leak into logs."""
        return (
            "Secrets(gemini_api_key='***', zotero_api_key='***', "
            f"zotero_user_id='{self.zotero_user_id}', "
            f"zotero_library_type='{self.zotero_library_type}', "
            f"openalex_email='{self.openalex_email}')"
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Fully resolved, typed configuration for the entire pipeline."""

    search: SearchConfig
    relevance: RelevanceConfig
    zotero: ZoteroConfig
    sources: SourcesConfig
    pipeline: PipelineSettings
    gemini: GeminiConfig
    ranker: RankerConfig
    secrets: Secrets
    project_root: Path


def _require_env(name: str) -> str:
    """Return an environment variable's value or raise a clear error.

    Args:
        name: The environment variable name.

    Returns:
        The variable's value.

    Raises:
        ConfigError: If the variable is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"See .env.example for how to obtain it."
        )
    return value


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Load and validate the full pipeline configuration.

    Loads environment variables from a ``.env`` file if present, validates that
    all required secrets exist, and parses ``config.yaml`` into typed dataclasses.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A fully populated :class:`PipelineConfig`.

    Raises:
        ConfigError: If a required env var is missing or the YAML is malformed.
    """
    load_dotenv()

    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". See .env.example for how to obtain each value."
        )

    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - passthrough
        raise ConfigError(f"Failed to parse {path}: {exc}") from exc

    search_raw = raw.get("search", {})
    relevance_raw = raw.get("relevance", {})
    zotero_raw = raw.get("zotero", {})
    sources_raw = raw.get("sources", {})
    pipeline_raw = raw.get("pipeline", {})
    gemini_raw = raw.get("gemini", {})
    ranker_raw = raw.get("ranker", {})

    interests = " ".join(str(relevance_raw.get("interests", "")).split())

    thresholds = [float(t) for t in ranker_raw.get("thresholds", [0.55, 0.4, 0.25, 0.1])]
    if len(thresholds) != 4:
        raise ConfigError(
            "ranker.thresholds must list exactly four descending cutoffs "
            "[t5, t4, t3, t2]."
        )

    return PipelineConfig(
        search=SearchConfig(
            queries=list(search_raw.get("queries", [])),
            days_back=int(search_raw.get("days_back", 7)),
            max_results_per_query=int(search_raw.get("max_results_per_query", 50)),
        ),
        relevance=RelevanceConfig(
            min_score=int(relevance_raw.get("min_score", 4)),
            interests=interests,
        ),
        zotero=ZoteroConfig(
            library_type=str(zotero_raw.get("library_type", "user")),
            inbox_collection=str(zotero_raw.get("inbox_collection", "AI Inbox")),
        ),
        sources=SourcesConfig(
            enable_semantic_scholar=bool(sources_raw.get("enable_semantic_scholar", False)),
            enable_arxiv=bool(sources_raw.get("enable_arxiv", False)),
            enable_crossref=bool(sources_raw.get("enable_crossref", False)),
        ),
        pipeline=PipelineSettings(
            log_to_file=bool(pipeline_raw.get("log_to_file", True)),
            log_file=str(pipeline_raw.get("log_file", "pipeline.log")),
            gemini_max_concurrency=int(pipeline_raw.get("gemini_max_concurrency", 4)),
        ),
        gemini=GeminiConfig(
            model=str(gemini_raw.get("model", "gemini-2.0-flash")),
            temperature=float(gemini_raw.get("temperature", 0.0)),
            enrich=bool(gemini_raw.get("enrich", True)),
        ),
        ranker=RankerConfig(thresholds=thresholds),
        secrets=Secrets(
            gemini_api_key=_require_env("GEMINI_API_KEY"),
            zotero_api_key=_require_env("ZOTERO_API_KEY"),
            zotero_user_id=_require_env("ZOTERO_USER_ID"),
            zotero_library_type=_require_env("ZOTERO_LIBRARY_TYPE"),
            openalex_email=_require_env("OPENALEX_EMAIL"),
        ),
        project_root=path.parent.resolve(),
    )
