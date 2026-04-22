"""
build_features.py — Precompute semantic search embeddings for ReadRadar.

Reads the processed book catalog, restricts it to the shared 5,000-book
sampled set, builds a combined text field, and produces aligned search
artifacts that src/search.py consumes.

Outputs:
    data/artifacts/search_books.parquet   — metadata for search result cards
    data/artifacts/search_embeddings.npy  — (N, 384) L2-normalized float32
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


BOOKS_PATH = Path("data/processed/books.parquet")
ARTIFACTS_DIR = Path("data/artifacts")

SEARCH_BOOKS_PATH = ARTIFACTS_DIR / "search_books.parquet"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "search_embeddings.npy"
SAMPLED_IDS_PATH = ARTIFACTS_DIR / "sampled_book_ids.json"

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 8
EMBED_DIM = 384
EXPECTED_BOOK_COUNT = 5_000


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

    if not SAMPLED_IDS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {SAMPLED_IDS_PATH}")

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

    with open(SAMPLED_IDS_PATH, "r") as f:
        sampled_ids = set(json.load(f))

    books_df["book_id"] = books_df["book_id"].astype(str)
    books_df = books_df[books_df["book_id"].isin(sampled_ids)].reset_index(drop=True)

    print(f"Books after filtering to sampled ids: {len(books_df)}")
    if len(books_df) != EXPECTED_BOOK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_BOOK_COUNT} rows after sampled-id filter, "
            f"got {len(books_df)}. Some sampled books are missing from "
            "books.parquet or have empty search_text."
        )

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    embeddings = _encode_documents(model, books_df["search_text"].tolist())

    if embeddings.shape[0] != len(books_df):
        raise RuntimeError(
            "Embedding row count does not match metadata row count; refusing to save."
        )

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