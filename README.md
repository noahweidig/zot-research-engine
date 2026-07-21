# zot-research-engine

Automated AI research discovery pipeline for Zotero.

## 1. What This Does

`zot-research-engine` automatically discovers newly published research papers,
uses Google Gemini to score how relevant each one is to *your* research
interests, and files the best ones into a Zotero collection — every single day,
with no manual effort. It searches [OpenAlex](https://openalex.org) (and
optionally arXiv and Crossref) for papers published in the last few days,
removes anything you have already seen or already have in your library, ranks
the remainder with a free local relevance filter, asks Gemini to rate the most
promising ones from 1 to 5, and adds the high scorers to a Zotero "AI Inbox"
collection tagged and summarized for quick triage.

To stay comfortably inside Google Gemini's free tier, the pipeline does the bulk
relevance work for free and locally: a [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
relevance ranker (no API key, no network, no quota — the same free
relevance-ranking technique open scholarly tools like Semantic Scholar and CORE
expose) scores every candidate against your interests first, and only the top
handful are sent to Gemini for the fine-grained 1–5 score. If Gemini is ever
unavailable (e.g. the free daily quota is exhausted), the pipeline falls back to
that free score so you still get a useful report instead of nothing.

The whole thing runs on a free GitHub Actions schedule. Fork the repo, edit one
YAML file, set four secrets, and you get a fresh curated literature review
committed to your repository (and pushed to Zotero) automatically. It is
designed so you never have to touch the Python code — all behavior is driven by
`config.yaml`.

## 2. Fork & Configure (Quick Start)

The entire setup takes under 20 minutes.

1. **Fork this repository** — click "Fork" at the top right of the GitHub page.

2. **Edit `config.yaml`** — this is the only file you need to change. Every field
   is documented in [Customization](#5-customization) below. At minimum, set:
   - `search.queries` — the topics to search for.
   - `relevance.interests` — a plain-language description of what you care about.
   - `relevance.min_score` — how strict to be (4 is a good default).

3. **Set GitHub Secrets** — in your fork go to
   **Settings → Secrets and variables → Actions → New repository secret** and add
   each of these:

   | Secret | Where to get it |
   |---|---|
   | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) → "Get API key" → "Create API key". Free. |
   | `ZOTERO_API_KEY` | [zotero.org/settings/keys/new](https://www.zotero.org/settings/keys/new) → enable **library access** and **write access**. |
   | `ZOTERO_USER_ID` | [zotero.org/settings/keys](https://www.zotero.org/settings/keys) → shown as "Your userID for use in API calls is …". |
   | `ZOTERO_LIBRARY_TYPE` | Almost always `user` (use `group` only for a group library). |
   | `OPENALEX_EMAIL` | Your email address — joins the faster OpenAlex "polite pool". |

4. **Enable GitHub Actions in your fork** — go to the **Actions** tab and click
   the button to enable workflows (forks have Actions disabled by default).

5. **Trigger a manual run to test** — Actions tab → **Daily Literature Review**
   → **Run workflow**. Watch the run; when it finishes, a new report appears
   under `reports/` and matching items appear in your Zotero "AI Inbox".

After that, the pipeline runs automatically every day at 08:00 UTC.

## 3. Running Locally

```bash
git clone https://github.com/your-username/zot-research-engine
cd zot-research-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
python main.py --dry-run
```

`--dry-run` runs the full pipeline (fetch, dedupe, rank, report) but does **not**
write anything to Zotero — perfect for testing your configuration. Remove the
flag to write for real.

Run the tests with:

```bash
pytest
```

## 4. CLI Reference

| Flag | Description |
|---|---|
| `--dry-run` | Run the full pipeline but skip writing to Zotero. |
| `--report-only` | Regenerate the Markdown report from the last JSON report, without fetching or ranking. |
| `--max-results N` | Override `search.max_results_per_query` from config. |
| `--min-score N` | Override `relevance.min_score` from config. |

## 5. Customization

All behavior lives in `config.yaml`:

**`search`**
- `queries` — list of search strings, run against every enabled source.
- `days_back` — only consider papers published within this many days.
- `max_results_per_query` — cap on results requested per query, per source.

**`relevance`**
- `min_score` — minimum Gemini score (1–5) required to be added to Zotero.
- `interests` — free-text description of your research focus. This drives the
  entire relevance judgement, so be specific.

**`zotero`**
- `library_type` — `user` or `group` (must match `ZOTERO_LIBRARY_TYPE`).
- `inbox_collection` — name of the collection new papers go into; created if
  missing.

**`sources`** — all optional sources are off by default and require no API key:
- `enable_semantic_scholar` — fill in missing abstracts via DOI lookup (not a
  search source).
- `enable_arxiv` — add arXiv preprints as an extra search source.
- `enable_crossref` — add Crossref published works as an extra search source.

**`pipeline`**
- `log_to_file` / `log_file` — also write logs to a file.
- `gemini_max_concurrency` — max simultaneous Gemini calls (protects your quota).

**`prefilter`** — the free local relevance stage that runs *before* Gemini so its
free-tier quota is never overwhelmed (the usual cause of `429` failures):
- `enabled` — turn the local BM25 pre-filter on/off (default `true`).
- `top_k` — maximum papers sent to Gemini per run. Lower it if you still hit
  quota limits; raise it to let Gemini see more candidates.
- `min_similarity` — drop papers whose normalized relevance (0–1, relative to the
  best match in the batch) is below this before they reach Gemini. `0.0` keeps
  the full `top_k`.

**`gemini`**
- `model` — Gemini model id (e.g. `gemini-2.0-flash`).
- `temperature` — keep at `0.0` for deterministic scoring.
- `requests_per_minute` — cap on Gemini requests per minute; keep at or below
  your tier's RPM limit (the free tier is ~15 RPM) to avoid rate-limit errors.
- `fallback_to_prefilter` — if Gemini is unavailable, score papers from the free
  local filter instead of dropping them (default `true`). Such papers are tagged
  `Heuristic Score` in the report and in Zotero.

Secrets are **never** stored in `config.yaml` — they come only from environment
variables / GitHub Secrets.

## 6. Understanding the Output

**Zotero tags.** Every added item is tagged with:
- `AI Suggested` — added by this pipeline.
- `Score N` — the Gemini relevance score (1–5).
- Plus any topic tags Gemini generated (e.g. `remote sensing`, `deep learning`).
- `Heuristic Score` — present only when Gemini was unavailable and the paper was
  scored by the free local relevance filter as a fallback.

**The report.** Each run writes two files to `reports/`:
- `YYYY-MM-DD.md` — a human-readable review: summary statistics, a detailed entry
  for every added paper (authors, journal, year, score, tags, reason, summary,
  DOI), and a table of ignored papers.
- `YYYY-MM-DD.json` — the full ranked paper list for programmatic use.

**The scoring scale.**

| Score | Meaning |
|---|---|
| 5 | Must read: directly addresses core research interests |
| 4 | Very relevant: closely related, high value |
| 3 | Interesting: tangentially related |
| 2 | Low relevance: minor overlap |
| 1 | Ignore: not relevant |

Only papers scoring at or above `min_score` are added to Zotero; everything else
is listed in the report's "Ignored Papers" table.

**State.** `state/seen_ids.json` records the DOIs and OpenAlex IDs of every paper
ever evaluated, so nothing is processed twice. The GitHub Action commits this
file (and the reports) back to your repository after each run so state persists.

## 7. Troubleshooting

**Papers are missing abstracts.** OpenAlex sometimes lacks an abstract; such
papers are skipped (and logged) because Gemini needs an abstract to score them.
Enable `sources.enable_semantic_scholar: true` to recover many abstracts via DOI
lookup.

**Gemini quota / rate-limit errors (429).** The pipeline is designed around the
free tier's low limits. Before Gemini is ever called, the free local BM25
pre-filter (`prefilter` block) narrows the candidate set to the most relevant
`prefilter.top_k` papers, and `gemini.requests_per_minute` throttles calls to
stay under your RPM cap. If you still hit limits, lower `prefilter.top_k`,
lower `gemini.requests_per_minute`, lower `pipeline.gemini_max_concurrency`,
reduce `search.max_results_per_query`, or shorten your query list. If Gemini is
completely exhausted and `gemini.fallback_to_prefilter` is on (the default), the
run still completes using the free local relevance score, and those papers are
tagged `Heuristic Score` so you can tell them apart from genuine Gemini scores.

**Zotero authentication errors.** Make sure `ZOTERO_API_KEY` has **write access**
enabled, that `ZOTERO_USER_ID` is the numeric ID (not your username), and that
`ZOTERO_LIBRARY_TYPE` matches your library (`user` for personal). A Zotero
failure is logged but never crashes the run — you still get a report.

**GitHub Actions permission error on the commit step.** The workflow declares
`permissions: contents: write`. If the push still fails, go to **Settings →
Actions → General → Workflow permissions** in your fork and select
**Read and write permissions**.

**No papers added.** Either nothing scored above `min_score` (try lowering it or
broadening `interests`), or everything was a known duplicate. Check the run logs
and the "Ignored Papers" table in the report.
