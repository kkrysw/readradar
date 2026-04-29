# ReadRadar

A book discovery Streamlit app built on the [UCSD Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (Goodreads snapshot, 2017). Users can search a 5,000-book catalog by theme, read LLM-generated controversy tags and an overall reader judgment for any book, add books to a favorites list, and receive recency-weighted recommendations.

## Final app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app has four pages:

1. **Thematic Search** — semantic search over the 5,000 sampled books. Each result card shows cover, rating, relevance score, top controversy tags, and a description preview. Clicking *Details* opens a detail modal with metadata, the controversy summary (`top_tags` + `overall_judgment`), and the full description. *+ Favorite* adds the book to the saved list.
2. **Favorites** — every saved book shown as a first-class card in insertion order (oldest → newest). Each card has the same *Details* modal access as search results and a *Remove* action for removing the book from favorites. This is the only page where favorites are managed.
3. **Recommendations** — top-N recommendations computed from the favorites list. Newer favorites weigh more (recency-weighted persona, see below). With no favorites yet, the page shows top-rated picks as a starting point.
4. **Reading Neighborhoods** — semantic clusters of the catalog. A drop-down lists 24 neighborhoods with human-readable labels built from cleaned Goodreads shelves; picking one shows its most central books as cards with the same *Details* modal access.

## Core algorithms (explicitly implemented)

Two course-covered algorithms are implemented from scratch — not wrapped via `sklearn` or equivalent:

1. **Cosine-similarity nearest-neighbor retrieval** (NumPy).
   - `src/search.py::_cosine_top_k` — query vector vs. every book vector.
   - `src/recommend.py::_cosine_scores` — recency-weighted persona vector vs. every book vector.
   - Book embeddings are L2-normalized at build time so cosine similarity reduces to a single matrix-vector dot product; top-k uses `np.argpartition`.

2. **Spherical k-means++ clustering** (NumPy).
   - `src/clustering.py::spherical_kmeans` with `kmeans_plus_plus_init`, cosine-similarity assignment, centroid update with L2 re-normalization, and robust empty-cluster recovery.
   - Used to build 24 semantic "Reading Neighborhoods" over the search embeddings. Goodreads `popular_shelves` are used *only after clustering* to generate human-readable neighborhood labels.

## Official pipeline

Raw source files, expected in `data/raw/`:

| File | Contents |
|---|---|
| `goodreads_books.json.gz` | Book metadata — titles, descriptions, ratings, shelves, authors |
| `goodreads_interactions.csv` | User–book star ratings and shelving events |
| `goodreads_reviews_dedup.json.gz` | Written reviews — text, rating, vote counts |

Official commands, in order:

```bash
# 1. Preprocess raw Goodreads data into processed parquets
python src/preprocess.py

# 2. Select the shared 5,000-book sampled catalog (book_id = global key)
python src/sample_books.py

# 3. Controversy pipeline (LLM-backed: top_tags + overall_judgment)
python src/controversy_prep.py
python src/controversy_run_llm.py      # requires ANTHROPIC_API_KEY in .env
python src/controversy_finalize.py

# 4. Search artifacts (Nomic v1.5, 384D, L2-normalized)
python scripts/build_features.py

# 5. Recommendation artifacts (MiniLM-L6-v2, 384D, L2-normalized)
python scripts/build_rec_embeddings.py

# 6. UI master cache (metadata + controversy merged for the Streamlit app)
python scripts/build_ui_cache.py

# 7. Reading Neighborhoods — spherical k-means++ clusters over search embeddings
python scripts/build_book_clusters.py

# 8. Launch the app
streamlit run app/app.py
```

## Pipeline stages in detail

### Preprocessing — `src/preprocess.py`

Converts three raw Goodreads files into clean Parquet tables used by every downstream module. Multiprocessed streaming JSON parser, Polars-accelerated CSV reader, and a cheap `book_id` regex pre-filter keep this manageable even on the full dataset.

- **Stage 1 — Books**: keeps books with `ratings_count ≥ 1000` and a non-empty title. One row per book with metadata fields (title, description, author, publisher, shelves, average rating, ISBN, image_url, etc.).
- **Stage 2 — Interactions**: explicit star ratings only, books from Stage 1, users with ≥10 rated interactions. Also writes `user_book_indices.json` with integer index maps.
- **Stage 3 — Reviews**: English reviews (hybrid `language_code` + `langid`/`langdetect` detection) for valid books with either a rating or a text.
- **Stage 4 — Validate**: cross-checks all three tables and writes `preprocessing_summary.json`.

Thresholds live in `src/config.py` (`MIN_BOOK_RATINGS`, `MIN_USER_RATINGS`, `PROTO_BOOK_COUNT`).

### Sampling — `src/sample_books.py`

Shared 5,000-book catalog, restricted to books present in all three processed tables and ranked by popularity signals with deterministic tie-breakers. The selected `book_id` list is the global join key used by every downstream module and by the app.

### Controversy — `src/controversy_prep.py`, `src/controversy_run_llm.py`, `src/controversy_finalize.py`

1. `controversy_prep.py` deterministically cleans and down-samples reviews per book (4 negative / 4 neutral / 4 positive, waterfall top-up) and builds a per-book LLM input parquet.
2. `controversy_run_llm.py` calls Claude once per book with structured tool output (`positive_aspects`, `negative_aspects`, `overall_judgment`, `top_tags`). Append-only JSONL checkpoints, retry/backoff on transient API errors, resume-safe.
3. `controversy_finalize.py` produces the single clean artifact the app reads: `data/artifacts/controversy_final.parquet` with `top_tags` and `overall_judgment` for each of the 5,000 books.

### Search / recommendation artifacts

- `scripts/build_features.py` encodes title + description + shelves with Nomic v1.5 → `search_books.parquet` + `search_embeddings.npy` (both aligned by row order).
- `scripts/build_rec_embeddings.py` encodes title + description + top reviews with MiniLM-L6-v2 → `rec_embeddings.npy` + `rec_embeddings_ids.json` (matched by row position).

### UI cache — `scripts/build_ui_cache.py`

Joins processed metadata with the final controversy artifact, normalizes `top_tags` to a Python list, validates 5,000 rows, and writes `ui_books_cache.parquet` — the single table the app loads for display.

## Official artifacts

Consumed by the app:

| File | Purpose |
|---|---|
| `data/artifacts/ui_books_cache.parquet` | Display metadata + `top_tags` + `overall_judgment` |
| `data/artifacts/search_books.parquet` + `search_embeddings.npy` | Semantic search |
| `data/artifacts/rec_embeddings.npy` + `rec_embeddings_ids.json` | Favorites-based recommendations |
| `data/artifacts/book_clusters.parquet` + `book_cluster_summary.json` | Reading Neighborhoods |
| `data/processed/books.parquet` | Canonical metadata for the recommendation metadata merge |

Produced by the pipeline but not read directly by the app (kept for auditability):

- `data/artifacts/sampled_book_ids.json`
- `data/artifacts/controversy_final.parquet` + `controversy_final_summary.json`
- `data/artifacts/controversy_prep/` and `data/artifacts/controversy_run/` intermediates

## Repository layout

```
src/
  config.py                 shared paths + thresholds
  preprocess.py             raw Goodreads → processed parquets
  sample_books.py           shared 5,000-book catalog
  controversy_prep.py       deterministic per-book review prep
  controversy_run_llm.py    Claude inference with resume/checkpoint
  controversy_finalize.py   clean final controversy artifact
  search.py                 semantic search (pure NumPy cosine top-k)
  recommend.py              favorites-based recommendation (pure NumPy cosine)
  clustering.py             spherical k-means++ (pure NumPy)

scripts/
  build_features.py         search embeddings
  build_rec_embeddings.py   recommendation embeddings
  build_ui_cache.py         UI master table
  build_book_clusters.py    Reading Neighborhoods artifact

app/
  app.py                    Streamlit app
  styles.py                 CSS

notebooks/
  goodreads_exploration.ipynb   exploratory only; not part of the pipeline
```

## Data sources

[UCSD Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (Goodreads snapshot, 2017) — Wan et al., 2018.

## Evaluation

Each backend module has a concrete objective. This section documents what the module does, what the main algorithm is, and how we verified the output matches the stated intent.

### Preprocessing
- **Objective.** Convert three raw Goodreads files into consistent, deduplicated parquet tables with explicit thresholds.
- **How we verify.** Stage 4 writes `preprocessing_summary.json` containing row counts, rating distributions, and cross-table consistency checks (orphan `book_id`s that appear in interactions/reviews but not in books are flagged). Thresholds live in `src/config.py` so their effect is inspectable.

### Sampling — shared 5,000-book catalog
- **Objective.** Pick exactly 5,000 books present in all three processed tables with enough review + interaction support, using deterministic tie-breakers so the list is reproducible.
- **Algorithm.** Three-way intersection of the processed tables, then `review_count ≥ 30` and `interaction_count ≥ 30`, then sort by `ratings_count` descending, `text_reviews_count` descending, with `book_id` ascending as final tie-breaker (mergesort is stable so the ordering is deterministic). Take the top 5,000.
- **How we verify.** `sample_books.py` asserts the final count is exactly 5,000, asserts subset containment in the three-way intersection, and writes `sampling_summary.json` with min/median/max stats of the selected set for spot-checking.

### Semantic search
- **Objective.** Given a free-text user query, return books whose content (title + description + popular shelves) is semantically closest to the query.
- **Algorithm.**
  - At build time, each book's combined text is encoded with [`nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). Nomic v1.5 is **natively 768-D**; it is a Matryoshka-trained model designed to be truncated. We apply layer normalization, truncate to the first 384 dimensions, and L2-normalize. Documents are prefixed with `search_document:` as Nomic requires.
  - At query time, the user's query is encoded with the `search_query:` prefix and put through the identical layer-norm → truncate-384 → L2-normalize pipeline so query and documents live in the same normalized space.
  - Retrieval is **pure-NumPy cosine top-k** (`src/search.py::_cosine_top_k`): because all vectors are L2-normalized, `scores = embeddings @ query_vec` is exactly cosine similarity, and top-k is `np.argpartition(-scores, k-1)[:k]` followed by a final argsort. No `sklearn.pairwise.cosine_similarity` wrapper; the algorithm is implemented explicitly.
- **How we verify.** We ran representative thematic queries (e.g. *"dark academia with found family"*, *"post-apocalyptic hope"*, *"slow burn Victorian romance"*) and spot-checked that the top-k match expected genres/themes, with scores degrading smoothly down the ranked list. Cosine scores are interpretable: values near 1 indicate strong alignment, values near the bottom of the top-100 indicate the natural relevance floor. Build-side assertions enforce exactly 5,000 rows with perfect row-order alignment between `search_books.parquet` and `search_embeddings.npy`.

### Favorites-based recommendation
- **Objective.** Given an ordered favorites list (oldest → newest), return the top-N books the user has not already liked, weighted so newer favorites influence the result more.
- **Algorithm.**
  - At build time, each book's title + description + top-10 most-upvoted reviews are encoded with `all-MiniLM-L6-v2` (natively **384-D**, L2-normalized at encode time). Output artifact shape: `(5000, 384)`.
  - At query time, recency weights are `np.linspace(1/(2n), 2/n, n)` normalized to sum to 1 — the newest weight is ≈ 4× the oldest. The persona vector is the weighted average of the liked books' embeddings; since weights sum to 1, this is a proper weighted average in the same embedding space.
  - The persona is L2-normalized once, then scored against all book embeddings with a single matrix-vector dot product (`src/recommend.py::_cosine_scores`). Already-liked books are filtered out; the remainder is sorted by similarity desc with average rating as a deterministic tie-breaker; the top N are returned.
- **How we verify.** We ran curated favorite lists end-to-end — e.g. three literary-fiction picks should yield literary-fiction recommendations; appending a sci-fi title as the newest favorite should noticeably shift the results toward sci-fi because the newest book dominates the persona. `compute_recency_weights` is deterministic; the sum-to-1 property and the newest/oldest ≈ 4 ratio hold by construction. Build-side assertions enforce exactly 5,000 aligned rows across `rec_embeddings.npy` and `rec_embeddings_ids.json`.

### Controversy — LLM-generated tags + overall judgment
- **Objective.** For each of the 5,000 books, produce an evidence-grounded reader-reception summary — 2–4 `top_tags` (UI-facing) plus a 2–3-sentence `overall_judgment` — that is defensible against the sampled reviews.
- **Algorithm.**
  1. **Deterministic review prep** (`controversy_prep.py`): strict English filter (langid + quality gates), minimum word count, dedupe by `review_id` and by (book, user, text). Per-book **balanced sampling** — 4 negative (1–2★) + 4 neutral (3★) + 4 positive (4–5★), with waterfall top-up so every book has exactly 12 reviews. Deterministic seed 42.
  2. **LLM call** (`controversy_run_llm.py`): one Claude Haiku 4.5 call per book, temperature 0.2, with:
     - the 12 sampled reviews,
     - objective rating statistics (mean, std dev, and the full rating distribution),
     - a prompt that forbids outside knowledge and explicitly instructs the model to reflect the actual distribution rather than force symmetry, and
     - a forced structured tool call returning `positive_aspects`, `negative_aspects`, `overall_judgment`, and `top_tags`.
     Outputs are validated (required fields, list lengths, tag count 2–4, returned `book_id` matches input), normalized, and append-only-checkpointed to JSONL. Transient API errors are retried with exponential backoff.
  3. **Finalize** (`controversy_finalize.py`): rebuilds `controversy_final.parquet` from the JSONL, asserts exactly 5,000 rows, asserts the book_id set matches the sampled catalog exactly, and asserts non-empty `top_tags` and `overall_judgment` for every row.
- **How we verify.** Schema assertions above rule out structural issues. For semantic fidelity, we manually audited ≈ 20–30 books by reading the 12 sampled reviews alongside the generated output and checking:
  - each tag is supported by recurring themes in the reviews,
  - `overall_judgment` reflects the dominant sentiment instead of forcing a balanced both-sides framing,
  - for books with sparse reviews, the model falls back to the conservative catch-all sentence rather than fabricating specifics.
  We additionally used stronger judge LLMs to cross-check a random sample of outputs against the underlying reviews for fidelity and hallucination.

### Reading Neighborhoods — spherical k-means++ clustering
- **Objective.** Group the 5,000 sampled books into ~24 semantic neighborhoods so users can browse the catalog by theme. Cluster labels must be readable without the noise of raw Goodreads shelves.
- **Algorithm.**
  - L2-normalize the search embeddings (they already are, but the function is applied defensively).
  - **k-means++ initialization** adapted to cosine distance: first centroid uniformly at random, each next centroid sampled with probability proportional to `(1 - max_cosine_sim_to_existing_centroids)^2`. If that distribution collapses, fall back to uniform over unchosen points.
  - **Lloyd iterations**: assign each book to its most similar centroid by cosine; recompute centroids as the mean of assigned vectors; L2-normalize back to the unit sphere. Any cluster that ends up empty is reinitialized to the point currently farthest from its own centroid. Stop when assignments stop changing, centroid shift falls below `tol`, or `max_iter` is hit.
  - **Labels from cleaned shelves** (only after clustering). `popular_shelves` strings are split and filtered through a deterministic pipeline: drop year/rating tags, utility shelves (`to-read`, `owned`, `dnf`, audiobook-formats, library shelves), too-short or non-ASCII-alphabetic tokens, and a small author/series blocklist; normalize near-duplicate variants (`sci-fi` → `science-fiction`, `graphic-novel`/`comics` → `graphic-novels`, `religious` → `religion`, `historical` → `historical-fiction`, …). Shelves are then counted with representative-weighted frequencies — cluster-central books contribute more (rank 1–10 × 3, rank 11–50 × 1.5, rest × 1) — and broad tags (`fiction`, `classic`, `adult`, …) are downweighted so specific shelves surface first. Each cluster is named by its top three shelves joined with ` · `.
  - **Representative books**: within each cluster, rank members by descending centroid similarity; top 5 marked `is_representative = True`.
- **How we verify.** `scripts/build_book_clusters.py` asserts 5,000 output rows, no duplicate `book_id`, no empty clusters, no NaN centroid similarities, that cluster sizes sum to 5,000, and — after the global uniqueness pass — no two clusters share the same label string or same token set in a different order. We additionally read the cluster summary JSON and spot-check labels and representative titles: *Vampires & Paranormal · Young Adult · Mystery & Thriller* reps include *Dracula* and the *Vampire Kisses* series; *Religion & Spirituality · Christian · Memoir & Biography* reps include *The Pilgrim's Progress* and *Searching for God Knows What*; *Mystery & Thriller · Crime & Detective · British Literature* reps are dominated by Hercule Poirot titles. The algorithm converges in under 30 iterations with a final cosine-distance inertia that is stable across reruns (same seed).

### Final UI cache
- **Objective.** Assemble the single parquet the app loads for every search card, recommendation card, and detail modal.
- **How we verify.** `build_ui_cache.py` inner-joins the final controversy artifact with the processed metadata on `book_id`, normalizes `top_tags` into a Python list, asserts exactly 5,000 rows, and fails loudly if any sampled book lacks metadata. The app only reads this table plus the pre-built search/recommendation artifacts; no live LLM call happens at runtime.
