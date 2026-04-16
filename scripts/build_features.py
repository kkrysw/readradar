"""
build_features.py - Precompute semantic embeddings for ReadRadar search.

Builds a searchable books table and embedding matrix from:
    data/proto/books.parquet

Outputs:
    data/artifacts/search_books.parquet
    data/artifacts/book_embeddings.npy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


BOOKS_PATH = Path("data/proto/books.parquet")
ARTIFACTS_DIR = Path("data/artifacts")

SEARCH_BOOKS_PATH = ARTIFACTS_DIR / "search_books.parquet"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "book_embeddings.npy"

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MAX_BOOKS = 2000
BATCH_SIZE = 4
EMBED_DIM = 128


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


def _encode_documents(model, texts):
    """
    Encode documents using Nomic retrieval prefix.
    Apply layer norm, truncate dimension, and L2 normalize.
    """
    prefixed = ["search_document: " + text for text in texts]

    embeddings = model.encode(
        prefixed,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :EMBED_DIM]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype(np.float32)


def build_search_artifacts():
    """Create semantic search artifacts."""
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

    print(f"Books after cleaning: {len(books_df)}")

    if len(books_df) > MAX_BOOKS:
        books_df = books_df.sample(n=MAX_BOOKS, random_state=42).reset_index(drop=True)

    print(f"Books used for embeddings: {len(books_df)}")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = _encode_documents(model, books_df["search_text"].tolist())

    books_df.to_parquet(SEARCH_BOOKS_PATH, index=False)
    np.save(EMBEDDINGS_PATH, embeddings)

    print("Saved search artifacts:")
    print(f"- {SEARCH_BOOKS_PATH}")
    print(f"- {EMBEDDINGS_PATH}")
    print(f"Embedding shape: {embeddings.shape}")


def main():
    """Run the feature-building pipeline."""
    build_search_artifacts()


if __name__ == "__main__":
    main()