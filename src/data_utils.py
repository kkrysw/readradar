"""
data_utils.py — ReadRadar shared data helpers
Provides consistent, memory-safe loading of processed Parquet files
and lightweight validation utilities used across the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports of config so data_utils can be used before config is finalized
# ---------------------------------------------------------------------------
def _cfg():
    try:
        import config as c
        return c
    except ImportError:
        raise ImportError(
            "config.py not found. Make sure src/ is on your PYTHONPATH "
            "or run from the repo root."
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_books(proto: bool = False) -> pd.DataFrame:
    """
    Load the processed books table.

    Args:
        proto: If True, load the lightweight proto dataset instead.

    Returns:
        DataFrame with one row per book.
    """
    c = _cfg()
    path = c.PROTO_BOOKS_PATH if proto else c.BOOKS_PATH
    _assert_exists(path, "books")
    log.info(f"Loading books from {path}")
    return pd.read_parquet(path)


def load_interactions(proto: bool = False) -> pd.DataFrame:
    """
    Load the processed interactions table.

    Args:
        proto: If True, load the lightweight proto dataset instead.

    Returns:
        DataFrame with columns: user_id, book_id, is_read, rating,
        is_reviewed, user_idx, book_idx.
    """
    c = _cfg()
    path = c.PROTO_INTERACTIONS_PATH if proto else c.INTERACTIONS_PATH
    _assert_exists(path, "interactions")
    log.info(f"Loading interactions from {path}")
    return pd.read_parquet(path)


def load_reviews(proto: bool = False) -> pd.DataFrame:
    """
    Load the processed reviews table.

    Args:
        proto: If True, load the lightweight proto dataset instead.

    Returns:
        DataFrame with columns: review_id, user_id, book_id, rating,
        review_text, n_votes, n_comments, date_added.
    """
    c = _cfg()
    path = c.PROTO_REVIEWS_PATH if proto else c.REVIEWS_PATH
    _assert_exists(path, "reviews")
    log.info(f"Loading reviews from {path}")
    return pd.read_parquet(path)


def load_user_book_indices() -> tuple[dict, dict]:
    """
    Load the user_id → int and book_id → int index mappings saved during
    preprocessing. These are required by recommend.py to build the sparse matrix.

    Returns:
        (user_index, book_index) — both are str → int dicts.
    """
    c = _cfg()
    _assert_exists(c.INDICES_PATH, "user_book_indices")
    with open(c.INDICES_PATH) as fh:
        data = json.load(fh)
    return data["user_index"], data["book_index"]


def load_summary() -> dict:
    """Load the preprocessing summary JSON."""
    c = _cfg()
    _assert_exists(c.SUMMARY_PATH, "preprocessing_summary")
    with open(c.SUMMARY_PATH) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_book_id(book_id: str, books: Optional[pd.DataFrame] = None) -> bool:
    """
    Return True if book_id exists in the processed books table.
    Pass books= to reuse an already-loaded DataFrame; otherwise loads from disk.
    """
    if books is None:
        books = load_books()
    return str(book_id) in set(books["book_id"].astype(str))


def get_book(book_id: str, books: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    Return metadata dict for a single book, or None if not found.
    """
    if books is None:
        books = load_books()
    row = books[books["book_id"] == str(book_id)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_reviews_for_book(
    book_id: str,
    reviews: Optional[pd.DataFrame] = None,
    min_votes: int = 0,
) -> pd.DataFrame:
    """
    Return all reviews for a given book_id.

    Args:
        book_id:   Target book.
        reviews:   Pre-loaded reviews DataFrame (avoids re-reading from disk).
        min_votes: Optionally filter to reviews with >= min_votes upvotes.

    Returns:
        Filtered DataFrame, sorted by n_votes descending.
    """
    if reviews is None:
        reviews = load_reviews()
    subset = reviews[reviews["book_id"] == str(book_id)]
    if min_votes > 0:
        subset = subset[subset["n_votes"] >= min_votes]
    return subset.sort_values("n_votes", ascending=False).reset_index(drop=True)


def rating_distribution(book_id: str, reviews: Optional[pd.DataFrame] = None) -> dict:
    """
    Return a dict mapping rating value (1–5) to count for a given book.
    Useful for controversy scoring and display.
    """
    book_reviews = get_reviews_for_book(book_id, reviews=reviews)
    dist = (
        book_reviews[book_reviews["rating"] > 0]["rating"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    # Ensure all keys 1–5 are present even if count is 0
    return {i: int(dist.get(i, 0)) for i in range(1, 6)}


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found at {path}. "
            "Have you run `python src/preprocess.py` yet?"
        )