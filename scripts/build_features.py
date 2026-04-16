"""
build_features.py - Precompute text features for ReadRadar search.

Builds a searchable books table and a TF-IDF embedding matrix from:
    data/processed/books.parquet

Outputs:
    data/artifacts/search_books.parquet
    data/artifacts/book_tfidf.npy
    data/artifacts/tfidf_vectorizer.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


BOOKS_PATH = Path("data/processed/books.parquet")
ARTIFACTS_DIR = Path("data/artifacts")

SEARCH_BOOKS_PATH = ARTIFACTS_DIR / "search_books.parquet"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "book_tfidf.npy"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"


def _clean_text(value):
    """Convert null values to empty strings and normalize text."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _build_search_text(df):
    """Combine title, description, and shelf tags into one text field."""
    df = df.copy()

    df["search_text"] = (
        df["title"].apply(_clean_text) + " "
        + df["description"].apply(_clean_text) + " "
        + df["popular_shelves"].apply(_clean_text)
    ).str.strip()

    return df


def build_search_artifacts():
    """Create TF-IDF features and save artifacts for search."""
    if not BOOKS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BOOKS_PATH}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    books_df = pd.read_parquet(BOOKS_PATH)

    keep_cols = [
        "book_id",
        "title",
        "description",
        "popular_shelves",
        "average_rating",
        "ratings_count",
        "text_reviews_count",
        "language_code",
    ]
    keep_cols = [col for col in keep_cols if col in books_df.columns]
    books_df = books_df[keep_cols].copy()

    books_df = _build_search_text(books_df)
    books_df = books_df[books_df["search_text"] != ""].copy()
    books_df = books_df.drop_duplicates(subset=["book_id"]).reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )

    tfidf_matrix = vectorizer.fit_transform(books_df["search_text"]).toarray().astype(np.float32)

    books_df.to_parquet(SEARCH_BOOKS_PATH, index=False)
    np.save(TFIDF_MATRIX_PATH, tfidf_matrix)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Saved search artifacts:")
    print(f"- {SEARCH_BOOKS_PATH}")
    print(f"- {TFIDF_MATRIX_PATH}")
    print(f"- {VECTORIZER_PATH}")
    print(f"Indexed books: {len(books_df)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


def main():
    """Run the feature-building pipeline."""
    build_search_artifacts()


if __name__ == "__main__":
    main()