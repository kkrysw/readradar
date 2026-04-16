"""
search.py - Search and nearest-neighbor retrieval logic for ReadRadar.

Loads precomputed TF-IDF features and supports:
    1. thematic_search(query)
    2. similar_books(book_id)

Uses cosine similarity with a custom top-k retrieval implementation.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ARTIFACTS_DIR = Path("data/artifacts")

SEARCH_BOOKS_PATH = ARTIFACTS_DIR / "search_books.parquet"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "book_tfidf.npy"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"


def _load_artifacts():
    """Load books table, TF-IDF matrix, and fitted vectorizer."""
    if not SEARCH_BOOKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SEARCH_BOOKS_PATH}. Run python scripts/build_features.py first."
        )
    if not TFIDF_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TFIDF_MATRIX_PATH}. Run python scripts/build_features.py first."
        )
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {VECTORIZER_PATH}. Run python scripts/build_features.py first."
        )

    books_df = pd.read_parquet(SEARCH_BOOKS_PATH)
    tfidf_matrix = np.load(TFIDF_MATRIX_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return books_df, tfidf_matrix, vectorizer


def _normalize_rows(matrix):
    """L2-normalize each row of a matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector):
    """L2-normalize a 1D vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _cosine_top_k(query_vec, doc_matrix, top_k=10):
    """
    Return indices and cosine scores of the top-k nearest neighbors.

    Assumes query_vec is 1D and doc_matrix is 2D.
    """
    if len(doc_matrix) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    top_k = min(top_k, len(doc_matrix))

    query_vec = _normalize_vector(query_vec)
    doc_matrix = _normalize_rows(doc_matrix)

    scores = doc_matrix @ query_vec

    candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    sorted_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

    return sorted_idx, scores[sorted_idx]


def thematic_search(query, top_k=10):
    """Return top-k books matching a free-text query."""
    books_df, tfidf_matrix, vectorizer = _load_artifacts()

    query = str(query).strip()
    if query == "":
        return books_df.head(0).copy()

    query_vec = vectorizer.transform([query]).toarray()[0].astype(np.float32)
    indices, scores = _cosine_top_k(query_vec, tfidf_matrix, top_k=top_k)

    result_df = books_df.iloc[indices].copy().reset_index(drop=True)
    result_df["score"] = scores

    cols = [
        "book_id",
        "title",
        "average_rating",
        "ratings_count",
        "text_reviews_count",
        "language_code",
        "score",
    ]
    cols = [col for col in cols if col in result_df.columns]

    return result_df[cols]


def similar_books(book_id, top_k=10):
    """Return top-k books most similar to the given book_id."""
    books_df, tfidf_matrix, _ = _load_artifacts()

    matches = books_df.index[books_df["book_id"].astype(str) == str(book_id)].tolist()
    if len(matches) == 0:
        raise ValueError(f"book_id {book_id} not found")

    row_idx = matches[0]
    query_vec = tfidf_matrix[row_idx]

    indices, scores = _cosine_top_k(query_vec, tfidf_matrix, top_k=top_k + 1)

    result_df = books_df.iloc[indices].copy()
    result_df["score"] = scores

    result_df = result_df[result_df["book_id"].astype(str) != str(book_id)]
    result_df = result_df.head(top_k).reset_index(drop=True)

    cols = [
        "book_id",
        "title",
        "average_rating",
        "ratings_count",
        "text_reviews_count",
        "language_code",
        "score",
    ]
    cols = [col for col in cols if col in result_df.columns]

    return result_df[cols]


def main():
    """Run a small search demo."""
    print("Sample thematic search:")
    print(thematic_search("fantasy magic wizard school", top_k=5))

    print("\nSample similar-books search:")
    books_df, _, _ = _load_artifacts()
    sample_book_id = books_df.iloc[0]["book_id"]
    print(similar_books(sample_book_id, top_k=5))

if __name__ == "__main__":
    main()
 