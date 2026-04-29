"""
recommend.py — ReadRadar favorites-based recommendation.

Builds a recency-weighted user persona from a list of liked book IDs using
precomputed 384D embeddings, then recommends the top-N most similar unread
books from the 5,000-book sampled catalog via pure NumPy cosine similarity.

Weighting scheme:
    Given n books in the input list (index 0 = oldest, index n-1 = newest):
        weights are linearly spaced from 1/(2n) to 2/n, normalized to sum to 1.
        Newest book has 4x the influence of the oldest.

Similarity:
    Book embeddings are L2-normalized at build time, so cosine similarity is a
    plain dot product. The persona is L2-normalized once per call so the
    resulting scores are valid cosine values in roughly [-1, 1] and directly
    comparable to the search module's similarity scores.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("data/artifacts")

IN_EMBEDDINGS = ARTIFACTS_DIR / "rec_embeddings.npy"
IN_BOOK_IDS   = ARTIFACTS_DIR / "rec_embeddings_ids.json"
IN_BOOKS      = PROCESSED_DIR / "books.parquet"

TOP_N = 5

log = logging.getLogger("recommend")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def load_artifacts() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Load precomputed embeddings, their matching book IDs, and metadata."""
    embeddings = np.load(IN_EMBEDDINGS)

    with open(IN_BOOK_IDS, "r") as f:
        book_ids = [str(bid) for bid in json.load(f)]

    books_df = pd.read_parquet(
        IN_BOOKS,
        columns=["book_id", "title", "average_rating", "ratings_count"],
    )
    books_df["book_id"] = books_df["book_id"].astype(str)
    books_df["average_rating"] = pd.to_numeric(
        books_df["average_rating"], errors="coerce"
    )
    books_df["ratings_count"] = (
        pd.to_numeric(books_df["ratings_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    sampled_ids = set(book_ids)
    books_df = books_df[books_df["book_id"].isin(sampled_ids)].copy()

    return embeddings, book_ids, books_df


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def compute_recency_weights(n: int) -> np.ndarray:
    """
    Return normalized recency weights of length n.

    Weight increases linearly from 1/(2n) at the oldest position to 2/n at
    the newest. Normalized so weights sum to 1. Newest weight ≈ 4x oldest.
    """
    if n == 1:
        return np.array([1.0])
    weights = np.linspace(1 / (2 * n), 2 / n, n)
    return weights / weights.sum()


def build_persona(
    liked_book_ids: list[str],
    embeddings: np.ndarray,
    book_ids: list[str],
) -> np.ndarray:
    """Recency-weighted average of the liked books' embedding vectors."""
    id_to_idx = {bid: i for i, bid in enumerate(book_ids)}

    matched_vecs: list[np.ndarray] = []
    for bid in liked_book_ids:
        bid = str(bid)
        if bid in id_to_idx:
            matched_vecs.append(embeddings[id_to_idx[bid]])
        else:
            log.warning("Book ID %s not found in embeddings, skipping.", bid)

    if not matched_vecs:
        raise ValueError("None of the provided book IDs were found in the sampled catalog.")

    weights = compute_recency_weights(len(matched_vecs))
    vectors = np.stack(matched_vecs)
    return (weights[:, np.newaxis] * vectors).sum(axis=0)


def _cosine_scores(persona: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Pure NumPy cosine similarity between a single persona vector and every
    book embedding. Book embeddings are already L2-normalized at build time,
    so we only need to normalize the persona once and take a dot product.
    """
    norm = np.linalg.norm(persona)
    if norm == 0:
        return np.zeros(embeddings.shape[0], dtype=np.float32)
    persona_unit = (persona / norm).astype(np.float32)
    return (embeddings.astype(np.float32) @ persona_unit)


def recommend(
    liked_book_ids: list[str],
    embeddings: np.ndarray,
    book_ids: list[str],
    books_df: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Generate top-N recommendations based on the favorites list.

    Returns a DataFrame with [rank, book_id, title, average_rating, similarity],
    already filtered to exclude the liked books themselves.
    """
    if not liked_book_ids:
        return pd.DataFrame(
            columns=["rank", "book_id", "title", "average_rating", "similarity"]
        )

    persona = build_persona(liked_book_ids, embeddings, book_ids)
    similarities = _cosine_scores(persona, embeddings)

    scored = pd.DataFrame({"book_id": book_ids, "similarity": similarities})
    scored["book_id"] = scored["book_id"].astype(str)

    liked_set = {str(bid) for bid in liked_book_ids}
    scored = scored[~scored["book_id"].isin(liked_set)].copy()

    scored = scored.merge(
        books_df[["book_id", "title", "average_rating", "ratings_count"]],
        on="book_id",
        how="left",
    )
    scored["average_rating"] = (
        pd.to_numeric(scored["average_rating"], errors="coerce").fillna(0)
    )

    scored = (
        scored.sort_values(
            ["similarity", "average_rating"],
            ascending=[False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    scored.insert(0, "rank", scored.index + 1)
    return scored[["rank", "book_id", "title", "average_rating", "similarity"]]
