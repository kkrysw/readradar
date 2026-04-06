"""
config.py — ReadRadar centralized configuration
All paths, thresholds, and constants live here so every module
imports from one place instead of hardcoding strings.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # repo root

# ------------------ß---------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROTO_DIR     = DATA_DIR / "proto"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Raw source files
# ---------------------------------------------------------------------------
RAW_BOOKS        = RAW_DIR / "goodreads_books.json.gz"
RAW_INTERACTIONS = RAW_DIR / "goodreads_interactions.csv"
RAW_REVIEWS      = RAW_DIR / "goodreads_reviews_dedup.json.gz"

# ---------------------------------------------------------------------------
# Processed files
# ---------------------------------------------------------------------------
BOOKS_PATH        = PROCESSED_DIR / "books.parquet"
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"
REVIEWS_PATH      = PROCESSED_DIR / "reviews.parquet"
INDICES_PATH      = PROCESSED_DIR / "user_book_indices.json"
SUMMARY_PATH      = PROCESSED_DIR / "preprocessing_summary.json"

# Proto variants (same names, different directory)
PROTO_BOOKS_PATH        = PROTO_DIR / "books.parquet"
PROTO_INTERACTIONS_PATH = PROTO_DIR / "interactions.parquet"
PROTO_REVIEWS_PATH      = PROTO_DIR / "reviews.parquet"

# ---------------------------------------------------------------------------
# Preprocessing thresholds
# ---------------------------------------------------------------------------
MIN_BOOK_RATINGS  = 1000  # minimum ratings_count to keep a book
MIN_USER_RATINGS  = 10    # minimum rated interactions to keep a user
PROTO_BOOK_COUNT  = 5_000 # number of books in the proto dataset

# ---------------------------------------------------------------------------
# Search artifacts
# ---------------------------------------------------------------------------
TFIDF_MATRIX_PATH    = ARTIFACTS_DIR / "tfidf_matrix.npz"
TFIDF_VOCAB_PATH     = ARTIFACTS_DIR / "tfidf_vocab.json"
EMBEDDINGS_PATH      = ARTIFACTS_DIR / "book_embeddings.npy"
EMBEDDING_INDEX_PATH = ARTIFACTS_DIR / "book_embedding_index.json"

# ---------------------------------------------------------------------------
# Recommendation artifacts
# ---------------------------------------------------------------------------
USER_FACTORS_PATH = ARTIFACTS_DIR / "user_factors.npy"
BOOK_FACTORS_PATH = ARTIFACTS_DIR / "book_factors.npy"
SVD_N_COMPONENTS  = 50   # latent dimensions for truncated SVD

# ---------------------------------------------------------------------------
# Controversy scoring
# ---------------------------------------------------------------------------
# Minimum number of ratings a book needs for a controversy score to be computed
MIN_RATINGS_FOR_CONTROVERSY = 50