# ReadRadar — Work Log

## Project Overview

ReadRadar is a book discovery web application with two core goals:
1. **Personalized search and recommendations** — help users find books that match their taste using vector similarity and (optionally) collaborative filtering
2. **Controversy insights** — explain why readers disagree about a book using rating distributions and review sentiment

**Data sources:**
- UCSD Book Graph (2017) — book metadata, ratings, and reviews scraped from Goodreads public shelves
- Open Library API — supplementary book descriptions and subjects (future)

---

## Progress

### Completed: Preprocessing (`src/preprocess.py`)

The preprocessing pipeline is fully implemented and optimized. It converts three raw Goodreads files into clean Parquet files that all downstream modules (search, recommendations, controversy scoring) read from.

---

## How Preprocessing Works

### Input

Three raw files expected in `data/raw/`:

| File | Contents |
|---|---|
| `goodreads_books.json.gz` | Book metadata — titles, descriptions, ratings, shelves, authors |
| `goodreads_interactions.csv` | User–book activity — star ratings and shelving events |
| `goodreads_reviews_dedup.json.gz` | Written reviews — text, rating, vote counts |

### Pipeline stages

**Stage 1 — Books**
Streams the books gzip file using parallel worker processes. Each batch of lines is dispatched to a worker that JSON-parses and filters. Keeps only books with ≥1,000 ratings and a non-empty title. Drops duplicate book IDs. Outputs one row per book with metadata fields: title, description, author, publisher, shelves, average rating, ISBN, etc.

**Stage 2 — Interactions**
Reads the 230M-row interactions CSV using polars (multi-threaded, significantly faster than pandas). Filters to explicit star ratings only (drops shelved-but-unrated rows), keeps only books that passed Stage 1, and drops users with fewer than 10 rated interactions. Also generates integer index mappings (`user_book_indices.json`) for each unique user and book ID — needed by the recommendation module to build a sparse matrix.

**Stage 3 — Reviews**
Streams the reviews gzip file using parallel worker processes. Each line is first checked against valid book IDs via a cheap regex (skipping ~99% of lines in proto mode without JSON parsing), then full-parsed, language-detected (English only, using `langid` with a `langdetect` fallback), and filtered. Keeps English reviews for valid books that have either a star rating or review text.

**Stage 4 — Validation**
Cross-checks consistency across all three output tables — flags book IDs that appear in interactions or reviews but not in the books table. Writes `preprocessing_summary.json` with row counts, rating distributions, and thresholds used.

### Output

| File | Description |
|---|---|
| `books.parquet` | One row per book with all metadata |
| `interactions.parquet` | User–book star ratings with integer indices |
| `reviews.parquet` | Written reviews, English only |
| `user_book_indices.json` | String ID → integer index maps (for SVD/sparse matrix) |
| `preprocessing_summary.json` | Counts, distributions, thresholds |

Output goes to `data/processed/` for a full run or `data/proto/` for a proto run. The two directories are fully separate — a proto run never touches `data/processed/`.

### How to run

```bash
# Proto pipeline — top 5,000 books only, writes to data/proto/, fast
python src/preprocess.py --proto

# Full pipeline — all books with 1,000+ ratings, writes to data/processed/
python src/preprocess.py

# Single stage only
python src/preprocess.py --stage books
python src/preprocess.py --stage interactions
python src/preprocess.py --stage reviews
python src/preprocess.py --stage validate
```

### Thresholds (set in `src/config.py`)

| Threshold | Value | Meaning |
|---|---|---|
| `MIN_BOOK_RATINGS` | 1,000 | Books with fewer ratings are dropped |
| `MIN_USER_RATINGS` | 10 | Users with fewer rated interactions are dropped |
| `PROTO_BOOK_COUNT` | 5,000 | Number of books in the proto dataset |

### Loading preprocessed data (downstream use)

All modules load data via `src/data_utils.py`:

```python
from data_utils import load_books, load_interactions, load_reviews

load_books()               # data/processed/books.parquet
load_books(proto=True)     # data/proto/books.parquet
load_interactions(proto=True)
load_reviews(proto=True)
```

---

## Next Steps

The preprocessed data feeds into three downstream modules, each corresponding to a core app feature:

### 1. Search (`src/search.py`)
**What it needs:** `books.parquet` (titles, descriptions, popular shelves)

**What to build:**
- Represent each book as a vector using TF-IDF over title + description + shelf tags
- Optionally add dense embeddings (e.g. sentence-transformers) for semantic search
- Build a cosine similarity index so users can search by theme, genre, or a book they already like
- Artifacts written to `data/artifacts/`: `tfidf_matrix.npz`, `tfidf_vocab.json`, `book_embeddings.npy`

### 2. Controversy Scoring (`src/controversy.py`)
**What it needs:** `reviews.parquet` (ratings per book)

**What to build:**
- For each book, compute a controversy score from the rating distribution
- Key signals: variance of ratings, proportion of extreme ratings (1-star and 5-star), bimodality
- Books with high variance and high extreme-rating proportion are most controversial
- Requires `MIN_RATINGS_FOR_CONTROVERSY = 50` (already set in `config.py`)
- Also: optionally extract representative positive/negative opinions from review text using lightweight NLP

### 3. Recommendations (`src/recommend.py`)
**What it needs:** `interactions.parquet` + `user_book_indices.json`

**What to build:**
- Model user–book interactions as a sparse matrix
- Apply truncated SVD to learn latent factors for users and books
- Use dot product of latent factors to score unseen books for a given user
- Note: SVD may be replaced with a simpler approach depending on project direction

### 4. App (`app/app.py`)
Connects the three modules above into a web interface. Users enter preferences or a book they like, receive recommendations, see controversy scores, and can read structured summaries of reader disagreement.

---

## Changes Made

### Threshold change
- `MIN_BOOK_RATINGS` raised from 500 → 1,000 in `src/config.py` to improve recommendation and controversy scoring quality by ensuring each book has sufficient rating data.

### Performance fixes — `src/preprocess.py`

**Fix 1 — Language detection library:**
Replaced `langdetect` with `langid`. `langid` is significantly faster (single matrix multiply per call vs. multiple random restarts). Initialized once at module load with a restricted language set. `langdetect` remains as a fallback.

**Fix 2 — Proto pre-filtering:**
`--proto` previously ran all stages on the full dataset and carved out the small dataset at the end. Now, `valid_book_ids` is narrowed to the top 5,000 books before Stages 2 and 3, so the book-ID check skips ~99% of rows before language detection is reached.

**Fix 3 — Regex pre-filter before JSON parsing (Stage 3):**
Added `_BOOK_ID_RE` regex that extracts `book_id` from the raw line before calling `json.loads`. For proto, this eliminates ~99% of JSON parsing in Stage 3. For the full pipeline, it skips a significant fraction of lines.

**Fix 4 — `orjson` for faster JSON parsing:**
Replaced stdlib `json.loads` with `orjson.loads` (C-based, 2–5x faster). Falls back to stdlib if `orjson` is not installed. `pip install orjson`.

**Fix 5 — Polars for Stage 2 CSV reading:**
Replaced pandas chunked CSV reading with `polars.scan_csv` (multi-threaded, lazy evaluation). Stage 2 now reads the 230M-row interactions CSV using all available CPU cores. Falls back to pandas if polars is not installed. `pip install polars`.

**Fix 6 — Multiprocessing for Stages 1 and 3:**
Both gzip-streaming stages now dispatch line batches to a `multiprocessing.Pool`. Worker functions (`_process_book_batch`, `_process_review_batch`) are defined at module level for pickling compatibility. The review worker pool is initialized once per worker with the valid book ID set (`_init_review_worker`) to avoid pickling it with every batch.

**Fix 7 — Larger gzip read buffer:**
Wrapped gzip file reads with an 8 MB `io.BufferedReader` to reduce I/O syscall overhead.

**Fix 8 — Separation of `data/proto/` and `data/processed/`:**
A `--proto` run previously wrote Stage 1 output to `data/processed/` as a side effect. Now Stage 1 streams books into memory without writing to disk when `--proto` is set, carves the top-5,000 subset in memory, and writes directly to `data/proto/`. `data/processed/` is never touched during a proto run.
