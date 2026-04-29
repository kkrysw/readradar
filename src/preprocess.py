"""
preprocess.py — ReadRadar data preprocessing pipeline
Processes UCSD Book Graph (2017) into clean Parquet files for downstream
search, recommendation, and controversy scoring.

Stages:
  1. Books metadata  — goodreads_books.json.gz       → processed/books.parquet
  2. Interactions    — goodreads_interactions.csv    → processed/interactions.parquet
  3. Reviews         — goodreads_reviews_dedup.json.gz → processed/reviews.parquet
  4. Join & validate — consistency checks + summary stats

Run:
  python src/preprocess.py               # full pipeline
  python src/preprocess.py --stage books # single stage
  python src/preprocess.py --proto       # proto dataset (5K books) → data/proto/ only

Thresholds (edit config.py to override):
  MIN_BOOK_RATINGS  = 1000
  MIN_USER_RATINGS  = 10
  PROTO_BOOK_COUNT  = 5_000

Dependencies:
  pip install orjson langid polars
  (all optional — graceful fallbacks exist for each)
"""

import argparse
import gzip
import io
import json
import logging
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional fast dependencies
# ---------------------------------------------------------------------------

# orjson: C-based JSON parser, 2-5x faster than stdlib.
try:
    from orjson import loads as _json_loads
except ImportError:
    from json import loads as _json_loads

# polars: multi-threaded DataFrame library, used for Stage 2 CSV reading.
try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

# langid: fast language classifier, used in the main process and workers.
try:
    import langid as _langid_mod
    _langid_mod.set_languages(["en", "fr", "de", "es", "it", "pt", "nl"])
    _langid_classify = _langid_mod.classify
except ImportError:
    _langid_classify = None

# ---------------------------------------------------------------------------
# Paths & constants (always sourced from src/config.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, PROCESSED_DIR, PROTO_DIR, ARTIFACTS_DIR
from config import MIN_BOOK_RATINGS, MIN_USER_RATINGS, PROTO_BOOK_COUNT

RAW_BOOKS        = RAW_DIR / "goodreads_books.json.gz"
RAW_INTERACTIONS = RAW_DIR / "goodreads_interactions.csv"
RAW_REVIEWS      = RAW_DIR / "goodreads_reviews_dedup.json.gz"

OUT_BOOKS        = PROCESSED_DIR / "books.parquet"
OUT_INTERACTIONS = PROCESSED_DIR / "interactions.parquet"
OUT_REVIEWS      = PROCESSED_DIR / "reviews.parquet"
OUT_SUMMARY      = PROCESSED_DIR / "preprocessing_summary.json"

PROTO_BOOKS        = PROTO_DIR / "books.parquet"
PROTO_INTERACTIONS = PROTO_DIR / "interactions.parquet"
PROTO_REVIEWS      = PROTO_DIR / "reviews.parquet"

LOG_EVERY  = 100_000   # lines between progress logs for JSONL streaming

# Regex to extract book_id from a raw JSON line without a full parse.
# Used in Stage 3 to skip json.loads for non-matching lines.
# Handles both string ("12345") and integer (12345) representations.
_BOOK_ID_RE = re.compile(r'"book_id":\s*"?(\d+)"?')

# Multiprocessing settings.
_BATCH_SIZE = 10_000                            # lines per worker batch
_N_WORKERS  = max(1, (os.cpu_count() or 2) - 1)  # leave one core for main process
_GZ_BUFSIZE = 8 * 1024 * 1024                  # 8 MB gzip read buffer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in [PROCESSED_DIR, PROTO_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _safe_int(val, default=0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _batched(iterable, n: int):
    """Yield successive n-sized lists from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _open_gz(path: Path):
    """Open a gzip file with a large read buffer for faster decompression."""
    return io.TextIOWrapper(
        io.BufferedReader(gzip.open(path, "rb"), buffer_size=_GZ_BUFSIZE),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Stage 1 — Books (multiprocessing worker)
# ---------------------------------------------------------------------------
def _process_book_batch(lines: list) -> tuple:
    """
    Parse and filter a batch of raw book JSON lines.
    Runs in a worker process.

    Returns:
        (records, skipped_ratings, skipped_no_title)
    """
    records = []
    skipped_ratings = skipped_no_title = 0

    for line in lines:
        try:
            obj = _json_loads(line)
        except Exception:
            continue

        ratings_count = _safe_int(obj.get("ratings_count", 0))
        if ratings_count < MIN_BOOK_RATINGS:
            skipped_ratings += 1
            continue

        title = (obj.get("title") or "").strip()
        if not title:
            skipped_no_title += 1
            continue

        shelves_raw = obj.get("popular_shelves") or []
        shelves = ",".join(
            s["name"] for s in shelves_raw
            if isinstance(s, dict) and s.get("name")
        )

        authors_raw = obj.get("authors") or []
        primary_author_id = (
            authors_raw[0].get("author_id", "") if authors_raw else ""
        )

        records.append({
            "book_id":              obj.get("book_id", ""),
            "work_id":              obj.get("work_id", ""),
            "title":                title,
            "title_without_series": (obj.get("title_without_series") or title).strip(),
            "description":          (obj.get("description") or "").strip(),
            "primary_author_id":    primary_author_id,
            "publisher":            (obj.get("publisher") or "").strip(),
            "publication_year":     _safe_int(obj.get("publication_year"), default=0),
            "num_pages":            _safe_int(obj.get("num_pages"), default=0),
            "format":               (obj.get("format") or "").strip(),
            "language_code":        (obj.get("language_code") or "").strip(),
            "is_ebook":             obj.get("is_ebook", "false") == "true",
            "average_rating":       _safe_float(obj.get("average_rating"), default=0.0),
            "ratings_count":        ratings_count,
            "text_reviews_count":   _safe_int(obj.get("text_reviews_count"), default=0),
            "popular_shelves":      shelves,
            "image_url":            (obj.get("image_url") or "").strip(),
            "isbn":                 (obj.get("isbn") or "").strip(),
            "isbn13":               (obj.get("isbn13") or "").strip(),
        })

    return records, skipped_ratings, skipped_no_title


# ---------------------------------------------------------------------------
# Stage 3 — Reviews (multiprocessing worker)
# ---------------------------------------------------------------------------
_w_book_ids: frozenset = frozenset()  # set once per worker via initializer
_w_classify = None                     # langid classify fn, set in initializer


def _init_review_worker(book_ids: frozenset):
    """
    Called once per worker process before any batches are processed.
    Sets worker-local copies of shared state so they don't need to be
    pickled with every batch.
    """
    global _w_book_ids, _w_classify
    _w_book_ids = book_ids
    try:
        import langid as _li
        _li.set_languages(["en", "fr", "de", "es", "it", "pt", "nl"])
        _w_classify = _li.classify
    except ImportError:
        _w_classify = None


def _detect_lang_worker(lang_code: str, text: str) -> bool:
    """Language detection using worker-local classifier."""
    if lang_code and lang_code.strip():
        code = lang_code.strip().lower()
        return code.startswith("en") or code == "eng"
    if not text or len(text.strip()) < 20:
        return False
    if _w_classify is not None:
        try:
            lang, _ = _w_classify(text)
            return lang == "en"
        except Exception:
            return False
    try:
        from langdetect import detect  # type: ignore
        return detect(text) == "en"
    except Exception:
        return False


def _process_review_batch(lines: list) -> tuple:
    """
    Parse and filter a batch of raw review JSON lines.
    Runs in a worker process; uses worker-local _w_book_ids and _w_classify.

    Returns:
        (records, skipped_book, skipped_lang, skipped_empty)
    """
    records = []
    skipped_book = skipped_lang = skipped_empty = 0

    for line in lines:
        # Cheap pre-filter: check book_id via regex before paying for json.loads.
        m = _BOOK_ID_RE.search(line)
        if not m or m.group(1) not in _w_book_ids:
            skipped_book += 1
            continue

        try:
            obj = _json_loads(line)
        except Exception:
            continue

        book_id = str(obj.get("book_id", ""))
        if book_id not in _w_book_ids:
            skipped_book += 1
            continue

        rating      = _safe_int(obj.get("rating"), default=0)
        review_text = (obj.get("review_text") or "").strip()

        if rating == 0 and not review_text:
            skipped_empty += 1
            continue

        lang_code = (obj.get("language_code") or "")
        if not _detect_lang_worker(lang_code, review_text):
            skipped_lang += 1
            continue

        records.append({
            "review_id":   str(obj.get("review_id", "")),
            "user_id":     str(obj.get("user_id", "")),
            "book_id":     book_id,
            "rating":      rating,
            "review_text": review_text,
            "n_votes":     _safe_int(obj.get("n_votes"), default=0),
            "n_comments":  _safe_int(obj.get("n_comments"), default=0),
            "date_added":  str(obj.get("date_added") or ""),
        })

    return records, skipped_book, skipped_lang, skipped_empty


# ---------------------------------------------------------------------------
# Stage 1 — Books
# ---------------------------------------------------------------------------
def process_books(output_path: Path = OUT_BOOKS) -> pd.DataFrame:
    """
    Stream goodreads_books.json.gz and parse in parallel worker batches.

    Filters:
      - ratings_count >= MIN_BOOK_RATINGS
      - must have a non-empty title

    Args:
        output_path: Where to write the parquet. Pass None to skip writing.
    """
    log.info("=== Stage 1: Books metadata ===")
    log.info(f"Source  : {RAW_BOOKS}")
    log.info(f"Filter  : ratings_count >= {MIN_BOOK_RATINGS}")
    log.info(f"Workers : {_N_WORKERS}")

    all_records = []
    total = skipped_ratings = skipped_no_title = 0

    with Pool(_N_WORKERS) as pool:
        with _open_gz(RAW_BOOKS) as f:
            for result in pool.imap_unordered(
                _process_book_batch, _batched(f, _BATCH_SIZE)
            ):
                recs, sr, snt = result
                all_records.extend(recs)
                skipped_ratings  += sr
                skipped_no_title += snt
                total            += len(recs) + sr + snt
                if total // LOG_EVERY > (total - len(recs) - sr - snt) // LOG_EVERY:
                    log.info(f"  Books read: ~{total:,}  kept: {len(all_records):,}")

    df = pd.DataFrame(all_records)

    # Compact dtypes — saves memory and disk space
    df["book_id"]            = df["book_id"].astype(str)
    df["work_id"]            = df["work_id"].astype(str)
    df["primary_author_id"]  = df["primary_author_id"].astype(str)
    df["ratings_count"]      = df["ratings_count"].astype(np.int32)
    df["text_reviews_count"] = df["text_reviews_count"].astype(np.int32)
    df["num_pages"]          = df["num_pages"].astype(np.int16)
    df["average_rating"]     = df["average_rating"].astype(np.float32)
    df["publication_year"]   = df["publication_year"].astype(np.int16)

    # Drop exact-duplicate book_ids (keep first)
    before_dedup = len(df)
    df = df.drop_duplicates(subset="book_id", keep="first").reset_index(drop=True)

    if output_path is not None:
        df.to_parquet(output_path, index=False)
        log.info(f"  Written → {output_path}")

    log.info(f"  Total lines read      : {total:,}")
    log.info(f"  Skipped (low ratings) : {skipped_ratings:,}")
    log.info(f"  Skipped (no title)    : {skipped_no_title:,}")
    log.info(f"  Duplicates dropped    : {before_dedup - len(df):,}")
    log.info(f"  Books kept            : {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Stage 2 — Interactions
# ---------------------------------------------------------------------------
def process_interactions(
    valid_book_ids: set,
    output_path: Path = OUT_INTERACTIONS,
    indices_path: Path = None,
) -> pd.DataFrame:
    """
    Read goodreads_interactions.csv and filter to valid books/users.

    Uses polars for multi-threaded CSV reading if available (pip install polars),
    otherwise falls back to pandas chunked reading.

    Filters:
      - rating > 0  (only explicit ratings; 0 means shelved but not rated)
      - book_id in valid_book_ids  (books that passed Stage 1)
      - users with >= MIN_USER_RATINGS interactions

    Args:
        output_path:  Where to write interactions parquet.
        indices_path: Where to write user_book_indices.json.
                      Defaults to same directory as output_path.
    """
    if indices_path is None:
        indices_path = output_path.parent / "user_book_indices.json"

    log.info("=== Stage 2: Interactions ===")
    log.info(f"Source  : {RAW_INTERACTIONS}")
    log.info(f"Filter  : rating > 0, book in valid set, user interactions >= {MIN_USER_RATINGS}")
    log.info(f"Backend : {'polars' if _HAS_POLARS else 'pandas (install polars for faster reads)'}")

    if _HAS_POLARS:
        lf = (
            pl.scan_csv(
                RAW_INTERACTIONS,
                schema_overrides={
                    "user_id":     pl.String,
                    "book_id":     pl.String,
                    "is_read":     pl.Int8,
                    "rating":      pl.Int8,
                    "is_reviewed": pl.Int8,
                },
            )
            .filter(pl.col("rating") > 0)
            .filter(pl.col("book_id").is_in(list(valid_book_ids)))
        )
        df_pl = lf.collect()
        log.info(f"  After rating + book filter : {len(df_pl):,} rows")

        # Filter: minimum user interactions
        user_counts = df_pl.group_by("user_id").agg(pl.len().alias("n"))
        valid_users = user_counts.filter(pl.col("n") >= MIN_USER_RATINGS)["user_id"]
        before = len(df_pl)
        df_pl = df_pl.filter(pl.col("user_id").is_in(valid_users))
        log.info(f"  Dropped {before - len(df_pl):,} rows from low-activity users")
        log.info(f"  Unique users : {df_pl['user_id'].n_unique():,}")
        log.info(f"  Unique books : {df_pl['book_id'].n_unique():,}")

        df = df_pl.to_pandas()

    else:
        # Fallback: pandas chunked reading
        chunks = []
        total_rows = skipped_book = skipped_no_rating = 0

        reader = pd.read_csv(
            RAW_INTERACTIONS,
            dtype={
                "user_id":     str,
                "book_id":     str,
                "is_read":     np.int8,
                "rating":      np.int8,
                "is_reviewed": np.int8,
            },
            chunksize=500_000,
        )

        for chunk in tqdm(reader, desc="Interactions chunks", unit="chunk"):
            total_rows += len(chunk)
            rated = chunk[chunk["rating"] > 0]
            skipped_no_rating += len(chunk) - len(rated)
            in_books = rated[rated["book_id"].isin(valid_book_ids)]
            skipped_book += len(rated) - len(in_books)
            if not in_books.empty:
                chunks.append(in_books)

        df = pd.concat(chunks, ignore_index=True)
        log.info(f"  After book + rating filter : {len(df):,} rows")

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
        before = len(df)
        df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)
        log.info(f"  Dropped {before - len(df):,} rows from low-activity users")
        log.info(f"  Unique users : {df['user_id'].nunique():,}")
        log.info(f"  Unique books : {df['book_id'].nunique():,}")
        log.info(f"  Total CSV rows read      : {total_rows:,}")
        log.info(f"  Skipped (no rating)      : {skipped_no_rating:,}")
        log.info(f"  Skipped (book not valid) : {skipped_book:,}")

    # Integer encoding — compact indices required by scipy sparse + sklearn SVD
    user_index = {uid: i for i, uid in enumerate(sorted(df["user_id"].unique()))}
    book_index = {bid: i for i, bid in enumerate(sorted(df["book_id"].unique()))}
    df["user_idx"] = df["user_id"].map(user_index).astype(np.int32)
    df["book_idx"] = df["book_id"].map(book_index).astype(np.int32)

    df.to_parquet(output_path, index=False)

    with open(indices_path, "w") as fh:
        json.dump({"user_index": user_index, "book_index": book_index}, fh)

    log.info(f"  Final interactions : {len(df):,}")
    log.info(f"  Written → {output_path}")
    log.info(f"  Written → {indices_path}")

    return df


# ---------------------------------------------------------------------------
# Stage 3 — Reviews
# ---------------------------------------------------------------------------
def process_reviews(valid_book_ids: set, output_path: Path = OUT_REVIEWS) -> pd.DataFrame:
    """
    Stream goodreads_reviews_dedup.json.gz and parse in parallel worker batches.

    Filters:
      - book_id in valid_book_ids  (cheap regex pre-check before json.loads)
      - English only (hybrid: language_code field, then langid/langdetect fallback)
      - drops rows with rating == 0 AND empty review_text

    Args:
        output_path: Where to write the parquet.
    """
    log.info("=== Stage 3: Reviews ===")
    log.info(f"Source  : {RAW_REVIEWS}")
    log.info(f"Filter  : English-only (hybrid), book in valid set")
    log.info(f"Workers : {_N_WORKERS}")

    all_records = []
    total = skipped_book = skipped_lang = skipped_empty = 0

    frozen_ids = frozenset(valid_book_ids)

    with Pool(
        _N_WORKERS,
        initializer=_init_review_worker,
        initargs=(frozen_ids,),
    ) as pool:
        with _open_gz(RAW_REVIEWS) as f:
            for result in pool.imap_unordered(
                _process_review_batch, _batched(f, _BATCH_SIZE)
            ):
                recs, sb, sl, se = result
                all_records.extend(recs)
                skipped_book  += sb
                skipped_lang  += sl
                skipped_empty += se
                batch_total    = len(recs) + sb + sl + se
                total         += batch_total
                if total // LOG_EVERY > (total - batch_total) // LOG_EVERY:
                    log.info(f"  Reviews read: ~{total:,}  kept: {len(all_records):,}")

    df = pd.DataFrame(all_records)
    df["rating"]     = df["rating"].astype(np.int8)
    df["n_votes"]    = df["n_votes"].astype(np.int16)
    df["n_comments"] = df["n_comments"].astype(np.int16)

    before_dedup = len(df)
    df = df.drop_duplicates(subset="review_id", keep="first").reset_index(drop=True)

    df.to_parquet(output_path, index=False)

    log.info(f"  Total lines read          : {total:,}")
    log.info(f"  Skipped (book not valid)  : {skipped_book:,}")
    log.info(f"  Skipped (non-English)     : {skipped_lang:,}")
    log.info(f"  Skipped (empty)           : {skipped_empty:,}")
    log.info(f"  Duplicates dropped        : {before_dedup - len(df):,}")
    log.info(f"  Reviews kept              : {len(df):,}")
    log.info(f"  Written → {output_path}")

    return df


# ---------------------------------------------------------------------------
# Stage 4 — Validation
# ---------------------------------------------------------------------------
def validate_and_summarize(
    books: pd.DataFrame,
    interactions: pd.DataFrame,
    reviews: pd.DataFrame,
    output_path: Path = OUT_SUMMARY,
) -> dict:
    """
    Cross-check consistency across all three tables.
    Writes a human-readable summary JSON to output_path.
    """
    log.info("=== Stage 4: Validation ===")

    books_ids        = set(books["book_id"])
    interaction_bids = set(interactions["book_id"])
    review_bids      = set(reviews["book_id"])

    books_with_interactions = len(books_ids & interaction_bids)
    books_with_reviews      = len(books_ids & review_bids)

    log.info(f"  Books in processed set             : {len(books_ids):,}")
    log.info(f"  Books with interactions            : {books_with_interactions:,}")
    log.info(f"  Books with reviews                 : {books_with_reviews:,}")

    orphan_interaction_books = interaction_bids - books_ids
    orphan_review_books      = review_bids - books_ids
    if orphan_interaction_books:
        log.warning(
            f"  {len(orphan_interaction_books):,} book_ids in interactions "
            "not in books table — these will be ignored by the pipeline."
        )
    if orphan_review_books:
        log.warning(
            f"  {len(orphan_review_books):,} book_ids in reviews "
            "not in books table — these will be ignored by the pipeline."
        )

    rating_counts = (
        reviews["rating"].value_counts().sort_index().to_dict()
    )

    summary = {
        "books_count":                 len(books),
        "interactions_count":          len(interactions),
        "reviews_count":               len(reviews),
        "unique_users_interactions":   int(interactions["user_id"].nunique()),
        "unique_users_reviews":        int(reviews["user_id"].nunique()),
        "books_with_interactions":     books_with_interactions,
        "books_with_reviews":          books_with_reviews,
        "rating_distribution_reviews": {str(k): int(v) for k, v in rating_counts.items()},
        "thresholds": {
            "min_book_ratings": MIN_BOOK_RATINGS,
            "min_user_ratings": MIN_USER_RATINGS,
        },
    }

    with open(output_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    log.info(f"  Summary written → {output_path}")
    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ReadRadar preprocessing pipeline")
    parser.add_argument(
        "--stage",
        choices=["books", "interactions", "reviews", "validate", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--proto",
        action="store_true",
        help="Scope pipeline to top 5K books; write output to data/proto/ only",
    )
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()

    # --- Stage 1 ---
    # In proto mode, skip writing to data/processed/ — parse into memory only,
    # then carve the top-N subset and write directly to data/proto/.
    if args.stage in ("books", "all"):
        books = process_books(output_path=None if args.proto else OUT_BOOKS)
    else:
        log.info("Loading existing books.parquet …")
        books = pd.read_parquet(OUT_BOOKS)

    # Route output paths and valid_book_ids based on --proto.
    # Proto writes all output to data/proto/ — data/processed/ is never touched.
    if args.proto:
        proto_books = books.nlargest(PROTO_BOOK_COUNT, "ratings_count").reset_index(drop=True)
        proto_books.to_parquet(PROTO_BOOKS, index=False)
        log.info(f"Proto books ({len(proto_books):,}) written → {PROTO_BOOKS}")
        valid_book_ids   = set(proto_books["book_id"].astype(str))
        out_interactions = PROTO_INTERACTIONS
        out_reviews      = PROTO_REVIEWS
        out_summary      = PROTO_DIR / "preprocessing_summary.json"
        out_indices      = PROTO_DIR / "user_book_indices.json"
        books_for_summary = proto_books
    else:
        valid_book_ids   = set(books["book_id"].astype(str))
        out_interactions = OUT_INTERACTIONS
        out_reviews      = OUT_REVIEWS
        out_summary      = OUT_SUMMARY
        out_indices      = PROCESSED_DIR / "user_book_indices.json"
        books_for_summary = books

    # --- Stage 2 ---
    if args.stage in ("interactions", "all"):
        interactions = process_interactions(
            valid_book_ids, output_path=out_interactions, indices_path=out_indices,
        )
    else:
        log.info("Loading existing interactions.parquet …")
        interactions = pd.read_parquet(out_interactions)

    # --- Stage 3 ---
    if args.stage in ("reviews", "all"):
        reviews = process_reviews(valid_book_ids, output_path=out_reviews)
    else:
        log.info("Loading existing reviews.parquet …")
        reviews = pd.read_parquet(out_reviews)

    # --- Stage 4 ---
    if args.stage in ("validate", "all"):
        validate_and_summarize(books_for_summary, interactions, reviews, output_path=out_summary)

    elapsed = time.time() - t0
    log.info(f"=== Done in {elapsed / 60:.1f} min ===")


if __name__ == "__main__":
    main()
