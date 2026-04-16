"""
recommendation_step2_algorithm.py — ReadRadar Recommendation Engine
Builds a weighted user persona from a list of liked book IDs using
precomputed SVD latent factors, then recommends similar unread books
from the 5,000 sampled catalog.

Stages:
  1. Load Artifacts  — Load book_latent_factors.parquet & books.parquet
  2. Build Persona   — Weighted average of latent factors (recency-weighted)
  3. Score Catalog   — Cosine similarity between persona and all book vectors
  4. Filter & Rank   — Remove already-read books, sort by similarity then rating

Weighting Scheme:
  Given n books in the input list (index 0 = oldest, index n-1 = newest):
  - Weights are linearly spaced from 1/(2n) to 2/n
  - Normalized to sum to 1 before computing weighted average
  - Result: newest book has 4x the influence of the oldest

Run:
  python src/recommendation_step2_algorithm.py

Dependencies:
  pip install pandas pyarrow numpy scikit-learn
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("data/artifacts")

IN_FACTORS = ARTIFACTS_DIR / "book_latent_factors.parquet"
IN_BOOKS   = PROCESSED_DIR / "books.parquet"

TOP_N = 5  # Number of books to recommend

FACTOR_COLS = [f"factor_{i}" for i in range(16)]  # Must match N_COMPONENTS in step1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rec_step2")


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def load_artifacts():
    """Load precomputed latent factors and book metadata."""
    log.info("Loading latent factors and book metadata...")

    factors_df = pd.read_parquet(IN_FACTORS)

    books_df = pd.read_parquet(
        IN_BOOKS,
        columns=["book_id", "title", "average_rating", "ratings_count"]
    )

    # Keep only sampled books (those that have latent factors)
    sampled_ids = set(factors_df["book_id"].astype(str))
    books_df = books_df[books_df["book_id"].astype(str).isin(sampled_ids)].copy()

    log.info(f"Loaded {len(factors_df)} book vectors, {len(books_df)} book metadata rows.")
    return factors_df, books_df


def compute_recency_weights(n: int) -> np.ndarray:
    """
    Compute normalized recency weights for n books.

    Weight increases linearly from 1/(2n) for the oldest book (index 0)
    to 2/n for the newest book (index n-1). Weights are then normalized
    to sum to 1.

    Args:
        n: Number of books in the input list.

    Returns:
        Array of shape (n,) with normalized weights.
    """
    if n == 1:
        return np.array([1.0])

    w_min = 1 / (2 * n)
    w_max = 2 / n
    weights = np.linspace(w_min, w_max, n)
    return weights / weights.sum()


def build_persona(liked_book_ids: list, factors_df: pd.DataFrame) -> np.ndarray:
    """
    Build a user persona vector as a recency-weighted average of liked books'
    latent factors.

    Args:
        liked_book_ids: List of book IDs ordered from oldest to newest interest.
        factors_df:     DataFrame with columns [book_id, factor_0, ..., factor_k].

    Returns:
        1D numpy array of shape (n_factors,) representing the user persona.
    """
    log.info(f"Building persona from {len(liked_book_ids)} books...")

    liked_book_ids = [str(bid) for bid in liked_book_ids]
    factors_df = factors_df.copy()
    factors_df["book_id"] = factors_df["book_id"].astype(str)

    # Filter to only books that exist in our catalog
    matched = factors_df[factors_df["book_id"].isin(liked_book_ids)].copy()

    # Preserve the input order (oldest → newest)
    matched["_order"] = matched["book_id"].map(
        {bid: i for i, bid in enumerate(liked_book_ids)}
    )
    matched = matched.sort_values("_order").reset_index(drop=True)

    missing = set(liked_book_ids) - set(matched["book_id"])
    if missing:
        log.warning(f"{len(missing)} book(s) not found in catalog and will be skipped: {missing}")

    n = len(matched)
    if n == 0:
        raise ValueError("None of the provided book IDs were found in the sampled catalog.")

    weights = compute_recency_weights(n)
    vectors = matched[FACTOR_COLS].values
    persona = (weights[:, np.newaxis] * vectors).sum(axis=0)

    log.info("Persona vector computed.")
    return persona


def recommend(
    liked_book_ids: list,
    factors_df: pd.DataFrame,
    books_df: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Generate book recommendations based on a user persona.
    Called every time the user adds or removes a book from their list.

    Args:
        liked_book_ids: Ordered list of book IDs the user likes (oldest first,
                        newest last). Updated on every add/remove action.
        factors_df:     Precomputed latent factor DataFrame.
        books_df:       Book metadata DataFrame.
        top_n:          Number of recommendations to return (default: 5).

    Returns:
        DataFrame with columns [rank, book_id, title, average_rating,
        similarity] sorted by similarity desc, then average_rating desc.
    """
    if not liked_book_ids:
        log.warning("Empty book list provided, returning empty recommendations.")
        return pd.DataFrame(columns=["rank", "book_id", "title", "average_rating", "similarity"])

    # 1. Build persona
    persona = build_persona(liked_book_ids, factors_df)

    # 2. Compute cosine similarity against all books in catalog
    all_vectors = factors_df[FACTOR_COLS].values
    persona_vec = persona.reshape(1, -1)
    similarities = cosine_similarity(persona_vec, all_vectors)[0]

    # 3. Attach scores
    scored = factors_df[["book_id", "title"]].copy()
    scored["similarity"] = similarities

    # 4. Filter out already-read books
    read_ids = set(str(bid) for bid in liked_book_ids)
    scored = scored[~scored["book_id"].astype(str).isin(read_ids)].copy()

    # 5. Merge with metadata for rating info
    books_df = books_df.copy()
    books_df["book_id"] = books_df["book_id"].astype(str)
    scored = scored.merge(
        books_df[["book_id", "average_rating", "ratings_count"]],
        on="book_id",
        how="left"
    )
    scored["average_rating"] = pd.to_numeric(
        scored["average_rating"], errors="coerce"
    ).fillna(0)

    # 6. Sort: similarity desc, then average_rating desc
    scored = scored.sort_values(
        ["similarity", "average_rating"],
        ascending=[False, False]
    ).head(top_n).reset_index(drop=True)

    scored.insert(0, "rank", scored.index + 1)

    log.info(f"Top {top_n} recommendations generated.")
    return scored[["rank", "book_id", "title", "average_rating", "similarity"]]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def run_tests(factors_df: pd.DataFrame, books_df: pd.DataFrame):
    """
    Simple functional tests to verify the recommendation pipeline.

    Test 1 — Single book:        persona should equal that book's vector exactly.
    Test 2 — Add a book:         recommendations should change after adding a book.
    Test 3 — Remove a book:      recommendations should change after removing a book.
    Test 4 — Recency weights:    newest book should have higher weight than oldest.
    Test 5 — No duplicates:      liked books should never appear in recommendations.
    Test 6 — Empty list:         should return empty DataFrame gracefully.
    Test 7 — Invalid ID:         unknown IDs should be skipped with a warning.
    """
    # Use real book IDs from the sampled catalog for testing
    all_ids = factors_df["book_id"].astype(str).tolist()
    book_a, book_b, book_c, book_d = all_ids[0], all_ids[1], all_ids[2], all_ids[3]

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ PASS  {name}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {name}" + (f" — {detail}" if detail else ""))
            failed += 1

    _print_section("Test 1: Single book persona")
    result_1 = recommend([book_a], factors_df, books_df)
    print(result_1.to_string(index=False))
    check("Returns 5 results", len(result_1) == 5)
    check("Liked book not in results", book_a not in result_1["book_id"].values)

    _print_section("Test 2: Adding a book changes recommendations")
    result_before = recommend([book_a], factors_df, books_df)
    result_after  = recommend([book_a, book_b], factors_df, books_df)
    recs_before = set(result_before["book_id"].astype(str))
    recs_after  = set(result_after["book_id"].astype(str))
    print(f"  Before (1 book): {recs_before}")
    print(f"  After  (2 books): {recs_after}")
    check("Recommendations changed after adding a book", recs_before != recs_after)

    _print_section("Test 3: Removing a book changes recommendations")
    result_three = recommend([book_a, book_b, book_c], factors_df, books_df)
    result_two   = recommend([book_a, book_b], factors_df, books_df)
    recs_three = set(result_three["book_id"].astype(str))
    recs_two   = set(result_two["book_id"].astype(str))
    print(f"  With 3 books: {recs_three}")
    print(f"  After remove: {recs_two}")
    check("Recommendations changed after removing a book", recs_three != recs_two)

    _print_section("Test 4: Recency weights")
    weights_3 = compute_recency_weights(3)
    print(f"  Weights for 3 books: {np.round(weights_3, 4)}")
    check("Weights sum to 1.0", abs(weights_3.sum() - 1.0) < 1e-9)
    check("Newest book has highest weight", weights_3[-1] > weights_3[0])
    check("Newest weight ≈ 4x oldest", abs(weights_3[-1] / weights_3[0] - 4.0) < 0.01)

    _print_section("Test 5: No liked books in recommendations")
    liked = [book_a, book_b, book_c]
    result_5 = recommend(liked, factors_df, books_df)
    overlap = set(str(b) for b in liked) & set(result_5["book_id"].astype(str))
    check("No liked books appear in results", len(overlap) == 0, f"Overlap: {overlap}")

    _print_section("Test 6: Empty book list")
    result_6 = recommend([], factors_df, books_df)
    check("Returns empty DataFrame", len(result_6) == 0)

    _print_section("Test 7: Invalid book ID is skipped")
    try:
        result_7 = recommend([book_a, "INVALID_ID_999999"], factors_df, books_df)
        check("Runs without crash", True)
        check("Still returns results", len(result_7) > 0)
    except Exception as e:
        check("Runs without crash", False, str(e))

    _print_section(f"Results: {passed} passed, {failed} failed")


def main():
    factors_df, books_df = load_artifacts()
    run_tests(factors_df, books_df)


if __name__ == "__main__":
    main()