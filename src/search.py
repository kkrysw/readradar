"""
search.py - Semantic search and nearest-neighbor retrieval logic for ReadRadar.

Loads precomputed embeddings and supports:
    1. thematic_search(query)
    2. similar_books(book_id)

Uses cosine similarity with a custom top-k retrieval implementation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


ARTIFACTS_DIR = Path("data/artifacts")

SEARCH_BOOKS_PATH = ARTIFACTS_DIR / "search_books.parquet"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "book_embeddings.npy"

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBED_DIM = 128


def _load_artifacts():
    """Load books table and embedding matrix."""
    if not SEARCH_BOOKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SEARCH_BOOKS_PATH}. Run python scripts/build_features.py first."
        )
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDINGS_PATH}. Run python scripts/build_features.py first."
        )

    books_df = pd.read_parquet(SEARCH_BOOKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    return books_df, embeddings


def _get_model():
    """Load embedding model."""
    return SentenceTransformer(MODEL_NAME)


def _encode_query(model, query):
    """
    Encode query using Nomic retrieval prefix.
    Apply layer norm, truncate dimension, and L2 normalize.
    """
    prefixed = ["search_query: " + str(query).strip()]

    embedding = model.encode(
        prefixed,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
    embedding = embedding[:, :EMBED_DIM]
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy()[0].astype(np.float32)


def _cosine_top_k(query_vec, doc_matrix, top_k=10):
    """
    Return indices and cosine scores of the top-k nearest neighbors.
    Assumes vectors are already normalized.
    """
    if len(doc_matrix) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    top_k = min(top_k, len(doc_matrix))

    scores = doc_matrix @ query_vec

    candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    sorted_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

    return sorted_idx, scores[sorted_idx]


def thematic_search(query, top_k=10):
    """Return top-k books matching a free-text query."""
    books_df, embeddings = _load_artifacts()

    query = str(query).strip()
    if query == "":
        return books_df.head(0).copy()

    model = _get_model()
    query_vec = _encode_query(model, query)

    indices, scores = _cosine_top_k(query_vec, embeddings, top_k=top_k)

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
    books_df, embeddings = _load_artifacts()

    matches = books_df.index[books_df["book_id"].astype(str) == str(book_id)].tolist()
    if len(matches) == 0:
        raise ValueError(f"book_id {book_id} not found")

    row_idx = matches[0]
    query_vec = embeddings[row_idx]

    indices, scores = _cosine_top_k(query_vec, embeddings, top_k=top_k + 1)

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


if __name__ == "__main__":
    print("=== Testing semantic search system ===")

    print("\n--- Thematic search: fantasy magic ---")
    results = thematic_search("fantasy magic", top_k=5)
    print(results[["title", "score"]])

    print("\n--- Similar books test ---")
    sample_book_id = results.iloc[0]["book_id"]
    similar = similar_books(sample_book_id, top_k=5)
    print(similar[["title", "score"]])