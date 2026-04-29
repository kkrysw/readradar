"""Build a shared 5,000-book sample for downstream modules.

The sample is restricted to books that appear in all three processed datasets
and ranked by popularity-first signals that support search, recommendations,
and controversy analysis.
"""

from __future__ import annotations

import json

import pandas as pd

import config as c

SAMPLE_SIZE = 5_000
MIN_REVIEW_COUNT = 30
MIN_INTERACTION_COUNT = 30
RANKING_CRITERIA = [
    {"column": "ratings_count", "order": "descending", "priority": 1},
    {"column": "text_reviews_count", "order": "descending", "priority": 2},
]


def load_processed_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three processed parquet files used by downstream modules."""
    books = pd.read_parquet(c.BOOKS_PATH)
    interactions = pd.read_parquet(c.INTERACTIONS_PATH)
    reviews = pd.read_parquet(c.REVIEWS_PATH)
    return books, interactions, reviews


def to_book_id_set(frame: pd.DataFrame) -> set[str]:
    """Return the unique book IDs from a DataFrame as strings."""
    return set(frame["book_id"].dropna().astype(str))


def build_sample(
    books: pd.DataFrame,
    interactions: pd.DataFrame,
    reviews: pd.DataFrame,
    sample_size: int = SAMPLE_SIZE,
) -> tuple[pd.DataFrame, int, int]:
    """Return the ranked sample plus the eligible pool size and intersection size."""
    books = books.copy()
    interactions = interactions.copy()
    reviews = reviews.copy()

    books["book_id"] = books["book_id"].astype(str)
    interactions["book_id"] = interactions["book_id"].astype(str)
    reviews["book_id"] = reviews["book_id"].astype(str)

    books_ids = to_book_id_set(books)
    interaction_ids = to_book_id_set(interactions)
    review_ids = to_book_id_set(reviews)
    eligible_ids = books_ids & interaction_ids & review_ids
    three_way_intersection_size = len(eligible_ids)

    if three_way_intersection_size < sample_size:
        raise ValueError(
            f"Only {three_way_intersection_size:,} books exist in the three-way intersection, "
            f"but {sample_size:,} are required."
        )

    interaction_counts = (
        interactions[interactions["book_id"].isin(eligible_ids)]
        .groupby("book_id", as_index=False)
        .size()
        .rename(columns={"size": "interaction_count"})
    )
    review_counts = (
        reviews[reviews["book_id"].isin(eligible_ids)]
        .groupby("book_id", as_index=False)
        .size()
        .rename(columns={"size": "review_count"})
    )

    ranked = (
        books[books["book_id"].isin(eligible_ids)]
        .merge(interaction_counts, on="book_id", how="left")
        .merge(review_counts, on="book_id", how="left")
    )

    ranked["interaction_count"] = ranked["interaction_count"].fillna(0).astype(int)
    ranked["review_count"] = ranked["review_count"].fillna(0).astype(int)

    eligible = ranked[
        (ranked["review_count"] >= MIN_REVIEW_COUNT)
        & (ranked["interaction_count"] >= MIN_INTERACTION_COUNT)
    ].copy()

    eligible_after_min_count_filters = len(eligible)

    if eligible_after_min_count_filters < sample_size:
        raise ValueError(
            f"Only {eligible_after_min_count_filters:,} books remain after applying "
            f"review_count >= {MIN_REVIEW_COUNT} and interaction_count >= {MIN_INTERACTION_COUNT}, "
            f"but {sample_size:,} are required."
        )

    eligible = eligible.sort_values(
        by=["ratings_count", "text_reviews_count", "book_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    sampled = eligible.head(sample_size).copy()

    if len(sampled) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size:,} sampled books, got {len(sampled):,}."
        )

    selected_ids = set(sampled["book_id"].astype(str))
    if not selected_ids.issubset(eligible_ids):
        raise RuntimeError("Selected books must exist in all three processed tables.")

    return sampled, eligible_after_min_count_filters, three_way_intersection_size


def summary_stats(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    """Return min/median/max summary statistics for a numeric column."""
    series = pd.to_numeric(frame[column], errors="coerce")
    return {
        "min": int(series.min()),
        "median": float(series.median()),
        "max": int(series.max()),
    }


def write_outputs(
    sampled: pd.DataFrame,
    eligible_after_min_count_filters: int,
    three_way_intersection_size: int,
) -> dict:
    """Write JSON and preview artifacts, then return the summary payload."""
    c.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    final_sampled = sampled.reset_index(drop=True).copy()
    sampled_ids = final_sampled["book_id"].astype(str).tolist()
    sampled_ids_path = c.ARTIFACTS_DIR / "sampled_book_ids.json"
    summary_path = c.ARTIFACTS_DIR / "sampling_summary.json"
    preview_path = c.ARTIFACTS_DIR / "sampled_books_preview.parquet"

    with sampled_ids_path.open("w") as fh:
        json.dump(sampled_ids, fh, indent=2)

    preview_columns = [
        "book_id",
        "title",
        "ratings_count",
        "text_reviews_count",
        "review_count",
        "interaction_count",
    ]
    available_columns = [column for column in preview_columns if column in final_sampled.columns]
    final_sampled.loc[:, available_columns].to_parquet(preview_path, index=False)

    if len(final_sampled) != len(sampled_ids):
        raise RuntimeError(
            "final_sampled_count does not match the number of sampled book IDs."
        )

    summary = {
        "three_way_intersection_size": int(three_way_intersection_size),
        "eligible_after_min_count_filters": int(eligible_after_min_count_filters),
        "final_sampled_count": int(len(final_sampled)),
        "min_review_count_threshold": MIN_REVIEW_COUNT,
        "min_interaction_count_threshold": MIN_INTERACTION_COUNT,
        "ranking_criteria": RANKING_CRITERIA,
        "selected_book_statistics": {
            "ratings_count": summary_stats(final_sampled, "ratings_count"),
            "text_reviews_count": summary_stats(final_sampled, "text_reviews_count"),
            "review_count": summary_stats(final_sampled, "review_count"),
            "interaction_count": summary_stats(final_sampled, "interaction_count"),
        },
    }

    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def main() -> None:
    """Create the shared 5,000-book sample and print a short summary."""
    books, interactions, reviews = load_processed_tables()
    sampled, eligible_after_min_count_filters, three_way_intersection_size = build_sample(
        books, interactions, reviews
    )
    summary = write_outputs(
        sampled,
        eligible_after_min_count_filters,
        three_way_intersection_size,
    )

    print("Sampled 5,000 books successfully.")
    print(f"Intersection size: {summary['three_way_intersection_size']:,}")
    print(
        "Eligible after minimum count filters: "
        f"{summary['eligible_after_min_count_filters']:,}"
    )
    print(f"Outputs written to: {c.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()