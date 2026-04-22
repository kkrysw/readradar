"""
build_rec_embeddings.py — ReadRadar recommendation embedding builder.

Reads the shared 5,000-book sampled catalog and generates recency-friendly
content embeddings that the app's recommend() function consumes.

Pipeline:
    1. Load sampled IDs and filter books.parquet + reviews.parquet.
    2. For each book, concatenate title + description + top-N most upvoted
       review texts into a single combined text field.
    3. Encode with all-MiniLM-L6-v2 → 384D vectors (L2-normalized).
    4. Save embeddings and matching book_id list, preserving row order.

Outputs:
    data/artifacts/rec_embeddings.npy        — (5000, 384) float32
    data/artifacts/rec_embeddings_ids.json   — list of book_ids aligned with rows
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("data/artifacts")

IN_BOOKS       = PROCESSED_DIR / "books.parquet"
IN_REVIEWS     = PROCESSED_DIR / "reviews.parquet"
IN_SAMPLED_IDS = ARTIFACTS_DIR / "sampled_book_ids.json"

OUT_EMBEDDINGS = ARTIFACTS_DIR / "rec_embeddings.npy"
OUT_BOOK_IDS   = ARTIFACTS_DIR / "rec_embeddings_ids.json"

# Hyperparameters
MAX_REVIEWS_PER_BOOK = 10     # top N upvoted reviews mixed into the text
EMBEDDING_BATCH_SIZE = 16
MODEL_NAME           = "all-MiniLM-L6-v2"
EXPECTED_BOOK_COUNT  = 5_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_rec_embeddings")


def load_and_filter_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sampled book IDs and filter books + reviews to the sampled set."""
    log.info("=== Stage 1: Loading Data ===")

    if not IN_SAMPLED_IDS.exists():
        raise FileNotFoundError(
            f"Missing {IN_SAMPLED_IDS}. Run `python src/sample_books.py` first."
        )

    with open(IN_SAMPLED_IDS, "r") as f:
        sampled_ids = {str(bid) for bid in json.load(f)}
    log.info("Loaded %d sampled book IDs.", len(sampled_ids))

    books_df = pd.read_parquet(IN_BOOKS, columns=["book_id", "title", "description"])
    books_df["book_id"] = books_df["book_id"].astype(str)
    books_df = books_df[books_df["book_id"].isin(sampled_ids)].copy()

    reviews_df = pd.read_parquet(
        IN_REVIEWS, columns=["book_id", "review_text", "n_votes"]
    )
    reviews_df["book_id"] = reviews_df["book_id"].astype(str)
    reviews_df = reviews_df[reviews_df["book_id"].isin(sampled_ids)].copy()

    log.info("Filtered to %d books and %d reviews.", len(books_df), len(reviews_df))
    return books_df, reviews_df


def aggregate_text(books_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Combine book description with top most-upvoted reviews into one text field."""
    log.info("=== Stage 2: Aggregating Text ===")

    reviews_df = reviews_df.sort_values(["book_id", "n_votes"], ascending=[True, False])
    top_reviews = (
        reviews_df.groupby("book_id")
        .head(MAX_REVIEWS_PER_BOOK)
        .groupby("book_id")["review_text"]
        .apply(lambda texts: " | User Review: ".join(texts))
        .reset_index()
        .rename(columns={"review_text": "aggregated_reviews"})
    )

    merged = pd.merge(books_df, top_reviews, on="book_id", how="left")
    merged["description"] = merged["description"].fillna("")
    merged["aggregated_reviews"] = merged["aggregated_reviews"].fillna("")

    merged["combined_text"] = (
        "Title: " + merged["title"] + ". "
        + "Description: " + merged["description"] + " "
        + "Reviews: " + merged["aggregated_reviews"]
    )
    return merged


def generate_embeddings(merged_df: pd.DataFrame) -> np.ndarray:
    """Encode the combined text into 384D L2-normalized embeddings."""
    log.info("=== Stage 3: Generating Embeddings ===")
    log.info("Loading model: %s", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        merged_df["combined_text"].tolist(),
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    log.info("Generated embeddings shape: %s", embeddings.shape)
    return np.asarray(embeddings, dtype=np.float32)


def save_outputs(embeddings: np.ndarray, merged_df: pd.DataFrame) -> None:
    """Persist embeddings and the aligned book_id list."""
    log.info("=== Stage 4: Saving Artifacts ===")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMBEDDINGS, embeddings)
    log.info("Saved embeddings to %s  shape=%s", OUT_EMBEDDINGS, embeddings.shape)

    book_ids = merged_df["book_id"].astype(str).tolist()
    with open(OUT_BOOK_IDS, "w") as f:
        json.dump(book_ids, f)
    log.info("Saved matching book IDs to %s", OUT_BOOK_IDS)


def main() -> None:
    t0 = time.time()

    books_df, reviews_df = load_and_filter_data()
    merged_df = aggregate_text(books_df, reviews_df)
    embeddings = generate_embeddings(merged_df)

    if len(merged_df) != EXPECTED_BOOK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_BOOK_COUNT} rows, got {len(merged_df)}. "
            "Some sampled book IDs are missing from books.parquet."
        )
    if embeddings.shape[0] != EXPECTED_BOOK_COUNT:
        raise RuntimeError(
            f"Embeddings row count {embeddings.shape[0]} does not match "
            f"expected {EXPECTED_BOOK_COUNT}."
        )

    save_outputs(embeddings, merged_df)
    log.info("=== Done in %.1f min ===", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
