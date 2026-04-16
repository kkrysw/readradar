"""
recommendation_step1_dataprocessing.py — ReadRadar Content Embedding Pipeline
Processes a sampled subset of 5,000 books, combining their descriptions
and top user reviews. Generates text embeddings using all-MiniLM-L6-v2 and
applies t-SNE to generate a 3D spatial map for visualization.

Stages:
  1. Data Loading     — Read sampled IDs, filter books.parquet & reviews.parquet
  2. Text Aggregation — Combine book description + top N most helpful reviews
  3. Embedding        — Generate 384D embeddings using all-MiniLM-L6-v2
  4. Save Embeddings  — Save 384D embeddings to artifacts/rec_embeddings.npy
  5. t-SNE Reduction  — Reduce 384D embeddings to 3D (x, y, z) coordinates
  6. Artifact Saving  — Save embeddings & 3D coordinates to artifacts/

Run:
  python src/recommendation_step1_dataprocessing.py

Thresholds & Hyperparameters:
  MAX_REVIEWS_PER_BOOK = 10  (Top 10 reviews sorted by n_votes)
  EMBEDDING_BATCH_SIZE = 16
  TSNE_PERPLEXITY      = 40

Dependencies:
  pip install pandas pyarrow scikit-learn sentence-transformers torch tqdm numpy

Outputs:
  data/artifacts/rec_embeddings.npy          — raw 384D embeddings for recommendation
  data/artifacts/rec_embeddings_ids.json     — matching book IDs for embeddings
  data/artifacts/read_universe_3d.parquet    — 3D coordinates for visualization
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("data/artifacts")

IN_BOOKS       = PROCESSED_DIR / "books.parquet"
IN_REVIEWS     = PROCESSED_DIR / "reviews.parquet"
IN_SAMPLED_IDS = ARTIFACTS_DIR / "sampled_book_ids.json"

OUT_EMBEDDINGS = ARTIFACTS_DIR / "rec_embeddings.npy"
OUT_BOOK_IDS   = ARTIFACTS_DIR / "rec_embeddings_ids.json"
OUT_UNIVERSE   = ARTIFACTS_DIR / "read_universe_3d.parquet"

# Hyperparameters
MAX_REVIEWS_PER_BOOK = 10
EMBEDDING_BATCH_SIZE = 16
MODEL_NAME           = "all-MiniLM-L6-v2"
TSNE_PERPLEXITY      = 40

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rec_step1")


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def load_and_filter_data():
    """Load sampled book IDs and filter the books and reviews dataframes."""
    log.info("=== Stage 1: Loading Data ===")

    with open(IN_SAMPLED_IDS, "r") as f:
        sampled_ids = set(str(bid) for bid in json.load(f))
    log.info(f"Loaded {len(sampled_ids)} sampled book IDs.")

    books_df = pd.read_parquet(IN_BOOKS, columns=["book_id", "title", "description"])
    books_df = books_df[books_df["book_id"].isin(sampled_ids)].copy()

    reviews_df = pd.read_parquet(
        IN_REVIEWS,
        columns=["book_id", "review_text", "n_votes"]
    )
    reviews_df = reviews_df[reviews_df["book_id"].isin(sampled_ids)].copy()

    log.info(f"Filtered to {len(books_df)} books and {len(reviews_df)} reviews.")
    return books_df, reviews_df


def aggregate_text(books_df, reviews_df):
    """Combine book description with the top most helpful reviews."""
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

    merged_df = pd.merge(books_df, top_reviews, on="book_id", how="left")
    merged_df["description"]        = merged_df["description"].fillna("")
    merged_df["aggregated_reviews"] = merged_df["aggregated_reviews"].fillna("")

    merged_df["combined_text"] = (
        "Title: " + merged_df["title"] + ". " +
        "Description: " + merged_df["description"] + " " +
        "Reviews: " + merged_df["aggregated_reviews"]
    )

    log.info("Text aggregation complete.")
    return merged_df


def generate_embeddings(merged_df):
    """Generate 384D embeddings using all-MiniLM-L6-v2."""
    log.info("=== Stage 3: Generating Embeddings ===")
    log.info(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    texts = merged_df["combined_text"].tolist()

    log.info("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    log.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings, merged_df):
    """Save 384D embeddings and matching book IDs to disk."""
    log.info("=== Stage 4: Saving Embeddings ===")

    np.save(OUT_EMBEDDINGS, embeddings)
    log.info(f"Saved embeddings to {OUT_EMBEDDINGS}  shape={embeddings.shape}")

    book_ids = merged_df["book_id"].astype(str).tolist()
    with open(OUT_BOOK_IDS, "w") as f:
        json.dump(book_ids, f)
    log.info(f"Saved matching book IDs to {OUT_BOOK_IDS}")


def perform_tsne(embeddings):
    """Reduce 384D embeddings to 3D coordinates via t-SNE for visualization."""
    log.info("=== Stage 5: t-SNE 3D Reduction ===")
    log.info("Note: t-SNE may take a few minutes to converge...")

    tsne = TSNE(
        n_components=3,
        perplexity=TSNE_PERPLEXITY,
        early_exaggeration=12,
        learning_rate="auto",
        init="pca",
        random_state=42,
        n_jobs=-1
    )

    coords_3d = tsne.fit_transform(embeddings)
    log.info(f"t-SNE complete. Output shape: {coords_3d.shape}")
    return coords_3d


def main():
    t0 = time.time()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    books_df, reviews_df = load_and_filter_data()

    # 2. Aggregate text
    merged_df = aggregate_text(books_df, reviews_df)

    # 3. Generate 384D embeddings
    embeddings = generate_embeddings(merged_df)

    # 4. Save embeddings for downstream recommendation use
    save_embeddings(embeddings, merged_df)

    # 5. t-SNE → 3D coordinates for visualization
    coords_3d = perform_tsne(embeddings)

    log.info("=== Stage 6: Saving Artifacts ===")
    universe_df = pd.DataFrame({
        "book_id": merged_df["book_id"].values,
        "title":   merged_df["title"].values,
        "x":       coords_3d[:, 0],
        "y":       coords_3d[:, 1],
        "z":       coords_3d[:, 2],
    })
    universe_df.to_parquet(OUT_UNIVERSE, index=False)
    log.info(f"Saved 3D universe to {OUT_UNIVERSE}")

    elapsed = time.time() - t0
    log.info(f"=== Pipeline completed in {elapsed / 60:.1f} minutes ===")


if __name__ == "__main__":
    main()