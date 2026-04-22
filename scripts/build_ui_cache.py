"""
build_ui_cache.py — Assemble the single parquet the Streamlit app loads.

Joins processed metadata with the final controversy artifact so the UI has
one tidy table containing:
    book_id, title, description, average_rating, ratings_count,
    text_reviews_count, image_url, publication_year, num_pages, publisher,
    language_code, top_tags, overall_judgment

Reads:
    data/processed/books.parquet
    data/artifacts/controversy_final.parquet

Writes:
    data/artifacts/ui_books_cache.parquet

Notes:
    `top_tags` in controversy_final may be stored as a list or a JSON string
    (depending on the run). This script normalizes it to a Python list before
    writing so the UI can read it uniformly.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR  = Path("data/processed")
ARTIFACTS_DIR  = Path("data/artifacts")
BOOKS_PATH     = PROCESSED_DIR / "books.parquet"
CONTRO_PATH    = ARTIFACTS_DIR / "controversy_final.parquet"
OUT_PATH       = ARTIFACTS_DIR / "ui_books_cache.parquet"

EXPECTED_BOOK_COUNT = 5_000

META_COLUMNS = [
    "book_id",
    "title",
    "description",
    "average_rating",
    "ratings_count",
    "image_url",
    "publication_year",
    "num_pages",
    "language_code",
    "publisher",
    "text_reviews_count",
]


def _normalize_tags(value) -> list[str]:
    """Coerce list/JSON-string/plain-string tag values into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return [raw]
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()]
    return [str(value).strip()]


def main() -> None:
    if not BOOKS_PATH.exists():
        raise FileNotFoundError(f"Missing {BOOKS_PATH}")
    if not CONTRO_PATH.exists():
        raise FileNotFoundError(f"Missing {CONTRO_PATH}")

    books = pd.read_parquet(BOOKS_PATH)
    books["book_id"] = books["book_id"].astype(str)
    books_meta = books[[c for c in META_COLUMNS if c in books.columns]].copy()

    contro = pd.read_parquet(CONTRO_PATH)
    contro["book_id"] = contro["book_id"].astype(str)
    contro_small = contro[["book_id", "top_tags", "overall_judgment"]].drop_duplicates("book_id")
    contro_small["top_tags"] = contro_small["top_tags"].apply(_normalize_tags)

    ui_cache = contro_small.merge(books_meta, on="book_id", how="left", sort=False)

    if len(ui_cache) != EXPECTED_BOOK_COUNT:
        raise RuntimeError(
            f"ui_books_cache has {len(ui_cache)} rows; expected {EXPECTED_BOOK_COUNT}."
        )
    if ui_cache["title"].isna().any():
        missing = int(ui_cache["title"].isna().sum())
        raise RuntimeError(
            f"{missing} books in controversy_final have no matching metadata in books.parquet."
        )

    column_order = ["book_id", "title", "top_tags", "overall_judgment"] + [
        c for c in META_COLUMNS if c not in {"book_id", "title"}
    ]
    ui_cache = ui_cache[[c for c in column_order if c in ui_cache.columns]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ui_cache.to_parquet(OUT_PATH, index=False)

    print(f"UI cache written: {OUT_PATH}  shape={ui_cache.shape}")


if __name__ == "__main__":
    main()
