"""
config.py — ReadRadar centralized configuration.

Canonical paths and thresholds for the preprocessing + sampling pipelines.
Downstream scripts (embedding builders, UI cache builder, Streamlit app)
use their own small path blocks for clarity; they are kept consistent with
the conventions defined here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # repo root

# ---------------------------------------------------------------------------
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
# Shared sample size for search / recommendation / controversy
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 5_000
