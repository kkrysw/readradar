# ReadRadar: AI Agent Instructions

## Project Overview
ReadRadar is an ML-powered literary catalog and discovery web application. It aggregates book metadata from three sources (Wikidata, Open Library, StoryGraph) into a unified SQLite database, then surfaces it via a Streamlit UI for theme-based book exploration.

## Architecture & Data Flow

### Pipeline: `ingest_catalog.py` (763 lines)
Five-stage ETL process producing SQLite + flattened CSV:

1. **Wikidata Seed** (SPARQL) → ~300-500 canonical books filtered by Project Gutenberg ID + Wikipedia sitelinks threshold
2. **Open Library Enrichment** → Title/author fuzzy-matching → work key, description, subjects, ISBN, covers
3. **StoryGraph Enrichment** (optional, scraping) → "vibe" tags extracted via BeautifulSoup regex patterns
4. **Deduplication** → Merges source records by composite key (OL work key → Gutenberg ID → ISBN13 → normalized title+author)
5. **SQLite Normalization** → Relational schema with works/authors/subjects/tags junction tables + flattened CSV export

**Key conventions:**
- All external HTTP requests use `cached_get_json()` / `cached_get_text()` with SHA256-hashed cache keys in `data/cache/`
- Conservative rate-limiting: WD_SLEEP=0.15s, OL_SLEEP=0.15s, SG_SLEEP=0.9s
- Fuzzy matching via `rapidfuzz.fuzz.token_set_ratio()` with hardcoded thresholds (e.g., OL doc score ≥0.60)

### UI: `streamlit_app.py` (159 lines)
Read-only catalog viewer with:
- SQL JOIN queries loading works + authors + subjects/tags (cached with `@st.cache_data`)
- Multi-field search (title/author/subjects/tags/description)
- Year range filter (best-effort; missing years retained)
- Optional StoryGraph-only filter
- Detail expanders showing full metadata

## Developer Workflows

### Run Pipeline
```bash
python ingest_catalog.py --out data/catalog.db --csv data/catalog_flat.csv --limit 300 --min-sitelinks 15 --storygraph 1
```
- `--limit`: Wikidata SPARQL result cap (smaller = faster for dev/testing)
- `--min-sitelinks`: Wikipedia presence threshold (higher = fewer but higher-quality seeds)
- `--storygraph 0`: Disable scraping if StoryGraph is unstable; still produces valid DB

### Run Web UI
```bash
streamlit run streamlit_app.py
```
Expected: `data/catalog.db` exists; set `CATALOG_DB` env var to override path.

### Dependencies
```
pandas requests beautifulsoup4 rapidfuzz SPARQLWrapper tqdm streamlit sqlite3
```

## Critical Code Patterns

### Deduplication Strategy (`_work_key()` → SHA1 prefix)
**Why:** Single source-of-truth for multi-source books. Records merged by completeness score (highest-quality wins, then field union/fill).
- Primary key chain: `ol_work_key` > `gutenberg_id` > `isbn13` > `norm(title+author)`
- Fallback normalization: lowercase, strip apostrophes, punctuation→space
- Used in: Lines 407–463 (deduplication), 485–494 (merge strategy)

### Fuzzy Matching Thresholds
**Why:** Exact string matching fails across sources (title variants, author abbreviations).
```python
# Open Library (line 220–226)
title_score = fuzz.token_set_ratio(title, t) / 100.0
author_score = fuzz.token_set_ratio(author, a) / 100.0
score = 0.65 * title_score + 0.30 * author_score - year_penalty
if best_score < 0.60: return None

# StoryGraph (line 283–285)
if best_score < 0.55: return None  # more lenient
```
Tuning these thresholds is the primary lever for precision/recall tradeoffs.

### Data Retention Strategy
- **Subjects:** Capped at 60 (line 278, prevents outlier explosion)
- **Tags:** Capped at 120 (line 370, StoryGraph scraping heuristic noise)
- **Description:** Capped at 4000 chars (line 280)
- These caps should be adjusted if ML downstream needs richer metadata

### Caching & Idempotence
- Cache keys: SHA256(url + sorted json.dumps(params))
- Semantics: `cached_get_json()` → reuses disk hit, sleeps *after* network (not on cache hit)
- Design: Disk cache avoids re-running expensive SPARQL/scraping on script reruns

## Integration Points

### External APIs (in order of reliance)
1. **Wikidata SPARQL** (line 151) → seed list; curated, stable
2. **Open Library Search/Work JSON** (lines 186–241) → primary enrichment; best coverage
3. **StoryGraph Browse/Book HTML** (lines 283–370) → optional sentiment/mood tags; breaks if site structure changes

### Database Schema (lines 505–560)
Normalized design:
- **works** (work_id PK) → scalar metadata + external IDs
- **work_authors, work_subjects, work_tags** → M:N relationships
- **record_map** → audit trail mapping source records to canonical works
- Streamlit queries use simple LEFT JOINs; no aggregation logic in DB (done in pandas)

## Common Tasks & Patterns

### Adding a New Enrichment Source
1. Write fetch function (cached): `def source_enrich_row(row: dict) -> dict`
2. Add to pipeline (line 715–718): Insert into `build_catalog()` loop
3. Extend work schema + CSV export (flatten_for_csv, line 686)
4. Add UI display in streamlit_app.py

### Debugging Match Failures
- Check `cached_get_json()` output in `data/cache/` (prefix: ol_search, ol_work, sg_browse, sg_book)
- Adjust fuzzy match thresholds or enable print() in match logic
- Run `--limit 10` for rapid iteration

### Modifying Deduplication Logic
- Edit `_work_key()` (line 400) for new key priority
- Edit `completeness()` (line 438) for merge scoring
- Test with `--limit 50` before full run

## File Dependencies
- **ingest_catalog.py** → (imports pandas, requests, BS4, rapidfuzz, SPARQLWrapper)
  - Produces: `data/catalog.db`, `data/catalog_flat.csv`, `data/cache/*.json`
- **streamlit_app.py** → (imports sqlite3, pandas, streamlit)
  - Reads: `data/catalog.db` (via `CATALOG_DB` env var)
- **data/catalog.db** → Shared state; must be generated before UI run
