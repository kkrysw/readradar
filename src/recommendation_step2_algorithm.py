"""
recommendation_step2_algorithm.py — ReadRadar Recommendation Engine
Builds a weighted user persona from a list of liked book IDs using
precomputed 384D embeddings, then recommends similar unread books
from the 5,000 sampled catalog using cosine similarity.

Stages:
  1. Load Artifacts  — Load rec_embeddings.npy & books.parquet
  2. Build Persona   — Weighted average of 384D embeddings (recency-weighted)
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

import json
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

IN_EMBEDDINGS = ARTIFACTS_DIR / "rec_embeddings.npy"
IN_BOOK_IDS   = ARTIFACTS_DIR / "rec_embeddings_ids.json"
IN_BOOKS      = PROCESSED_DIR / "books.parquet"

TOP_N = 5

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
    """Load precomputed 384D embeddings and book metadata."""
    log.info("Loading embeddings and book metadata...")

    embeddings = np.load(IN_EMBEDDINGS)

    with open(IN_BOOK_IDS, "r") as f:
        book_ids = json.load(f)

    books_df = pd.read_parquet(
        IN_BOOKS,
        columns=["book_id", "title", "average_rating", "ratings_count"]
    )
    books_df["book_id"] = books_df["book_id"].astype(str)
    books_df["average_rating"] = pd.to_numeric(books_df["average_rating"], errors="coerce")
    books_df["ratings_count"] = pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0).astype(int)

    # Keep only sampled books
    sampled_ids = set(book_ids)
    books_df = books_df[books_df["book_id"].isin(sampled_ids)].copy()

    log.info(f"Loaded embeddings shape: {embeddings.shape}, {len(books_df)} books.")
    return embeddings, book_ids, books_df


def compute_recency_weights(n: int) -> np.ndarray:
    """
    Compute normalized recency weights for n books.

    Weight increases linearly from 1/(2n) for the oldest book (index 0)
    to 2/n for the newest book (index n-1). Normalized to sum to 1.

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


def build_persona(liked_book_ids: list, embeddings: np.ndarray, book_ids: list) -> np.ndarray:
    """
    Build a user persona vector as a recency-weighted average of liked books'
    384D embeddings.

    Args:
        liked_book_ids: List of book IDs ordered from oldest to newest interest.
        embeddings:     2D array of shape (N, 384).
        book_ids:       List of book IDs matching the embedding rows.

    Returns:
        1D numpy array of shape (384,) representing the user persona.
    """
    log.info(f"Building persona from {len(liked_book_ids)} books...")

    liked_book_ids = [str(bid) for bid in liked_book_ids]
    id_to_idx = {bid: i for i, bid in enumerate(book_ids)}

    matched_ids = []
    matched_vecs = []
    for bid in liked_book_ids:
        if bid in id_to_idx:
            matched_ids.append(bid)
            matched_vecs.append(embeddings[id_to_idx[bid]])
        else:
            log.warning(f"Book ID {bid} not found in embeddings, skipping.")

    n = len(matched_vecs)
    if n == 0:
        raise ValueError("None of the provided book IDs were found in the sampled catalog.")

    weights = compute_recency_weights(n)
    vectors = np.stack(matched_vecs)                          # shape: (n, 384)
    persona = (weights[:, np.newaxis] * vectors).sum(axis=0)  # shape: (384,)

    log.info("Persona vector computed.")
    return persona


def recommend(
    liked_book_ids: list,
    embeddings: np.ndarray,
    book_ids: list,
    books_df: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Generate book recommendations based on a user persona.
    Called every time the user adds or removes a book from their list.

    Args:
        liked_book_ids: Ordered list of book IDs the user likes (oldest first,
                        newest last). Updated on every add/remove action.
        embeddings:     Precomputed 384D embedding matrix, shape (N, 384).
        book_ids:       List of book IDs matching embedding rows.
        books_df:       Book metadata DataFrame.
        top_n:          Number of recommendations to return (default: 5).

    Returns:
        DataFrame with columns [rank, book_id, title, average_rating, similarity].
    """
    if not liked_book_ids:
        log.warning("Empty book list provided, returning empty recommendations.")
        return pd.DataFrame(columns=["rank", "book_id", "title", "average_rating", "similarity"])

    # 1. Build persona
    persona = build_persona(liked_book_ids, embeddings, book_ids)

    # 2. Cosine similarity against all embeddings
    persona_vec = persona.reshape(1, -1)                          # shape: (1, 384)
    similarities = cosine_similarity(persona_vec, embeddings)[0]  # shape: (N,)

    # 3. Build scored DataFrame aligned to book_ids order
    scored = pd.DataFrame({
        "book_id":    book_ids,
        "similarity": similarities,
    })
    scored["book_id"] = scored["book_id"].astype(str)

    # 4. Filter out already-read books
    read_ids = set(str(bid) for bid in liked_book_ids)
    scored = scored[~scored["book_id"].isin(read_ids)].copy()

    # 5. Merge with metadata
    scored = scored.merge(
        books_df[["book_id", "title", "average_rating", "ratings_count"]],
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


def run_tests(embeddings, book_ids, books_df):
    """
    Simple functional tests to verify the recommendation pipeline.

    Test 1 — Single book:     returns 5 results, liked book not in results.
    Test 2 — Add a book:      recommendations change after adding a book.
    Test 3 — Remove a book:   recommendations change after removing a book.
    Test 4 — Recency weights: newest book has higher weight than oldest.
    Test 5 — No duplicates:   liked books never appear in recommendations.
    Test 6 — Empty list:      returns empty DataFrame gracefully.
    Test 7 — Invalid ID:      unknown IDs are skipped with a warning.
    """
    all_ids = book_ids
    book_a, book_b, book_c, book_d = all_ids[0], all_ids[1], all_ids[2], all_ids[3]

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ PASS  {name}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {name}" + (f" — {detail}" if detail else ""))
            failed += 1

    _print_section("Test 1: Single book persona")
    result_1 = recommend([book_a], embeddings, book_ids, books_df)
    print(result_1.to_string(index=False))
    check("Returns 5 results", len(result_1) == 5)
    check("Liked book not in results", book_a not in result_1["book_id"].values)

    _print_section("Test 2: Adding a book changes recommendations")
    result_before = recommend([book_a], embeddings, book_ids, books_df)
    result_after  = recommend([book_a, book_b], embeddings, book_ids, books_df)
    check("Recommendations changed", set(result_before["book_id"]) != set(result_after["book_id"]))

    _print_section("Test 3: Removing a book changes recommendations")
    result_three = recommend([book_a, book_b, book_c], embeddings, book_ids, books_df)
    result_two   = recommend([book_a, book_b], embeddings, book_ids, books_df)
    check("Recommendations changed", set(result_three["book_id"]) != set(result_two["book_id"]))

    _print_section("Test 4: Recency weights")
    weights_3 = compute_recency_weights(3)
    print(f"  Weights for 3 books: {np.round(weights_3, 4)}")
    check("Weights sum to 1.0", abs(weights_3.sum() - 1.0) < 1e-9)
    check("Newest book has highest weight", weights_3[-1] > weights_3[0])
    check("Newest weight ≈ 4x oldest", abs(weights_3[-1] / weights_3[0] - 4.0) < 0.01)

    _print_section("Test 5: No liked books in recommendations")
    liked = [book_a, book_b, book_c]
    result_5 = recommend(liked, embeddings, book_ids, books_df)
    overlap = set(str(b) for b in liked) & set(result_5["book_id"].astype(str))
    check("No liked books in results", len(overlap) == 0, f"Overlap: {overlap}")

    _print_section("Test 6: Empty book list")
    result_6 = recommend([], embeddings, book_ids, books_df)
    check("Returns empty DataFrame", len(result_6) == 0)

    _print_section("Test 7: Invalid book ID is skipped")
    try:
        result_7 = recommend([book_a, "INVALID_ID_999999"], embeddings, book_ids, books_df)
        check("Runs without crash", True)
        check("Still returns results", len(result_7) > 0)
    except Exception as e:
        check("Runs without crash", False, str(e))

    _print_section(f"Results: {passed} passed, {failed} failed")


def main():
    embeddings, book_ids, books_df = load_artifacts()
    run_tests(embeddings, book_ids, books_df)


if __name__ == "__main__":
    main()