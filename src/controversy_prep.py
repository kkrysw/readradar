"""Deterministic controversy preparation pipeline for LLM input generation.

Run from repo root:
	python src/controversy_prep.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import langid
import numpy as np
import pandas as pd


EXPECTED_BOOK_COUNT = 5000
MIN_WORD_COUNT = 10
RANDOM_SEED = 42
REVIEWS_PER_BOOK = 12
QUOTA_PER_BUCKET = 4

NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"
BUCKET_ORDER = [NEGATIVE, NEUTRAL, POSITIVE]


def repo_root() -> Path:
	"""Return repository root based on this file location."""
	return Path(__file__).resolve().parent.parent


def load_sampled_book_ids(path: Path) -> list[str]:
	"""Load sampled book IDs, normalize to strings, and validate expected size."""
	if not path.exists():
		raise FileNotFoundError(f"Missing sampled book IDs file: {path}")

	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		raise ValueError(f"Expected sampled book IDs as a JSON list in {path}")

	sampled_ids = [str(x).strip() for x in data if str(x).strip()]
	sampled_ids = list(dict.fromkeys(sampled_ids))

	if len(sampled_ids) != EXPECTED_BOOK_COUNT:
		raise ValueError(
			f"Expected {EXPECTED_BOOK_COUNT} sampled books, found {len(sampled_ids)}"
		)

	return sampled_ids


def load_required_parquet(path: Path, required_columns: set[str]) -> pd.DataFrame:
	"""Load parquet and validate that required columns exist."""
	if not path.exists():
		raise FileNotFoundError(f"Missing parquet file: {path}")

	df = pd.read_parquet(path)
	missing = sorted(required_columns.difference(df.columns))
	if missing:
		raise ValueError(f"Missing required columns in {path.name}: {', '.join(missing)}")
	return df


def word_count(text: Any) -> int:
	"""Simple whitespace-based word count after stripping."""
	if not isinstance(text, str):
		return 0
	stripped = text.strip()
	if not stripped:
		return 0
	return len(stripped.split())


def _passes_text_quality_gate(text: str) -> bool:
	"""Cheap gate before language detection to reject noisy short text."""
	stripped = text.strip()
	if len(stripped) < 20:
		return False

	alpha_chars = sum(1 for c in stripped if c.isalpha())
	if alpha_chars < 15:
		return False

	latin_chars = sum(1 for c in stripped if ("a" <= c <= "z") or ("A" <= c <= "Z"))
	if latin_chars / alpha_chars < 0.85:
		return False

	alpha_tokens = [tok for tok in stripped.split() if any(ch.isalpha() for ch in tok)]
	if len(alpha_tokens) < 5:
		return False

	return True


def is_likely_english(text: Any) -> bool:
	"""Conservative English detector using langid after basic quality checks."""
	if not isinstance(text, str):
		return False
	if not _passes_text_quality_gate(text):
		return False

	try:
		lang, _ = langid.classify(text.strip())
		return lang == "en"
	except Exception:
		return False


def _review_count_stats(clean_df: pd.DataFrame, sampled_ids: list[str]) -> dict[str, float | int]:
	"""Compute review count stats per sampled book after cleaning."""
	counts_series = clean_df.groupby("book_id").size()
	counts = [int(counts_series.get(book_id, 0)) for book_id in sampled_ids]
	counts_pd = pd.Series(counts)
	return {
		"total_valid_reviews_left": int(sum(counts)),
		"median_reviews_per_book_after_cleaning": float(counts_pd.median()),
		"min_reviews_for_a_single_book_after_cleaning": int(counts_pd.min()),
		"max_reviews_for_a_single_book_after_cleaning": int(counts_pd.max()),
	}


def clean_reviews(
	reviews_df: pd.DataFrame,
	sampled_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, float | int]]:
	"""Run deterministic controversy cleaning pipeline in the required order."""
	sampled_set = set(sampled_ids)
	working = reviews_df.copy()

	working["book_id"] = working["book_id"].astype("string").str.strip()

	drop_counts: dict[str, int] = {}

	in_sample_mask = working["book_id"].isin(sampled_set)
	drop_counts["dropped_because_book_id_not_in_sampled_list"] = int((~in_sample_mask).sum())
	working = working.loc[in_sample_mask].copy()

	rating_numeric = pd.to_numeric(working["rating"], errors="coerce")
	invalid_rating_mask = rating_numeric.isna() | (rating_numeric <= 0)
	drop_counts["dropped_because_rating_lte_0"] = int(invalid_rating_mask.sum())
	working = working.loc[~invalid_rating_mask].copy()
	working["rating"] = rating_numeric.loc[~invalid_rating_mask]

	stripped_text = working["review_text"].astype("string").str.strip()
	empty_text_mask = stripped_text.isna() | (stripped_text == "")
	drop_counts["dropped_because_empty_review_text"] = int(empty_text_mask.sum())
	working = working.loc[~empty_text_mask].copy()
	working["review_text"] = stripped_text.loc[~empty_text_mask]

	before = len(working)
	working = working.drop_duplicates(subset=["review_id"], keep="first").copy()
	drop_counts["dropped_because_duplicate_review_id"] = int(before - len(working))

	before = len(working)
	working = working.drop_duplicates(
		subset=["book_id", "user_id", "review_text"], keep="first"
	).copy()
	drop_counts["dropped_because_duplicate_book_user_review_text"] = int(before - len(working))

	english_mask = working["review_text"].map(is_likely_english)
	drop_counts["dropped_because_not_english"] = int((~english_mask).sum())
	working = working.loc[english_mask].copy()

	working["review_word_count"] = working["review_text"].map(word_count)
	short_mask = working["review_word_count"] < MIN_WORD_COUNT
	drop_counts["dropped_because_below_minimum_word_count"] = int(short_mask.sum())
	working = working.loc[~short_mask].copy()

	count_stats = _review_count_stats(working, sampled_ids)
	return working, drop_counts, count_stats


def assign_bucket_labels(df: pd.DataFrame) -> pd.DataFrame:
	"""Assign rating buckets used for balanced review sampling."""
	out = df.copy()
	ratings = pd.to_numeric(out["rating"], errors="coerce")
	if ratings.isna().any():
		raise ValueError("Found invalid ratings during bucket assignment")

	bucket = pd.Series(pd.NA, index=out.index, dtype="string")
	bucket = bucket.mask(ratings.isin([1, 2]), NEGATIVE)
	bucket = bucket.mask(ratings == 3, NEUTRAL)
	bucket = bucket.mask(ratings.isin([4, 5]), POSITIVE)

	if bucket.isna().any():
		raise ValueError("Found ratings outside expected 1-5 range during bucket assignment")

	out["bucket_label"] = bucket
	return out


def sample_reviews_for_book(
	book_df: pd.DataFrame,
	rng: np.random.Generator,
) -> pd.DataFrame:
	"""Sample one book with 4/4/4 target and waterfall top-up to 12 total."""
	selected_idx: set[Any] = set()
	selected_parts: list[pd.DataFrame] = []

	for label in BUCKET_ORDER:
		bucket_rows = book_df.loc[book_df["bucket_label"] == label]
		bucket_idx = bucket_rows.index.to_numpy()
		take_n = min(QUOTA_PER_BUCKET, len(bucket_idx))
		if take_n > 0:
			chosen = rng.choice(bucket_idx, size=take_n, replace=False)
			selected_idx.update(chosen.tolist())
			selected_parts.append(book_df.loc[chosen])

	sampled_initial = (
		pd.concat(selected_parts, axis=0, ignore_index=False)
		if selected_parts
		else pd.DataFrame(columns=book_df.columns)
	)

	shortfall = REVIEWS_PER_BOOK - len(sampled_initial)
	if shortfall > 0:
		residual = book_df.loc[~book_df.index.isin(selected_idx)]
		residual_idx = residual.index.to_numpy()
		if len(residual_idx) < shortfall:
			book_id = str(book_df["book_id"].iloc[0])
			raise ValueError(
				f"Book {book_id} cannot be filled to {REVIEWS_PER_BOOK} reviews. "
				f"Shortfall={shortfall}, residual_available={len(residual_idx)}"
			)
		top_up_idx = rng.choice(residual_idx, size=shortfall, replace=False)
		sampled = pd.concat([sampled_initial, book_df.loc[top_up_idx]], axis=0, ignore_index=False)
	else:
		sampled = sampled_initial

	shuffle_order = rng.permutation(len(sampled))
	sampled = sampled.iloc[shuffle_order].copy()
	return sampled


def sample_reviews_all_books(clean_df: pd.DataFrame, sampled_ids: list[str]) -> pd.DataFrame:
	"""Sample all books deterministically and validate exact per-book totals."""
	rng = np.random.default_rng(RANDOM_SEED)
	grouped = clean_df.groupby("book_id", sort=False)

	sampled_parts: list[pd.DataFrame] = []
	for book_id in sampled_ids:
		if book_id not in grouped.groups:
			raise ValueError(f"Sampled book {book_id} has no cleaned reviews available")
		book_df = grouped.get_group(book_id)
		sampled_parts.append(sample_reviews_for_book(book_df, rng))

	sampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)

	counts = sampled_df.groupby("book_id").size()
	bad_counts = counts[counts != REVIEWS_PER_BOOK]
	if not bad_counts.empty:
		example = {str(k): int(v) for k, v in bad_counts.head(10).items()}
		raise ValueError(
			f"Sampling validation failed; not all books have {REVIEWS_PER_BOOK} reviews: {example}"
		)

	expected_total = EXPECTED_BOOK_COUNT * REVIEWS_PER_BOOK
	if len(sampled_df) != expected_total:
		raise ValueError(
			f"Sampling total mismatch; expected {expected_total}, got {len(sampled_df)}"
		)

	return sampled_df[["book_id", "review_text"]].copy()


def _format_percentage(value: float) -> str:
	"""Format percentage as whole number when exact, otherwise one decimal place."""
	rounded = round(value)
	if abs(value - rounded) < 1e-9:
		return f"{rounded}%"
	return f"{value:.1f}%"


def format_rating_distribution_text(counts: dict[str, int]) -> str:
	"""Build rating distribution text from rating count dictionary (keys 1..5)."""
	total = sum(counts.values())
	if total <= 0:
		raise ValueError("Cannot format rating distribution with zero total ratings")

	parts = []
	for star in ["1", "2", "3", "4", "5"]:
		pct = (counts[star] / total) * 100.0
		parts.append(f"{star}★ {_format_percentage(pct)}")
	return "Rating distribution: " + ", ".join(parts)


def compute_rating_stats(interactions_df: pd.DataFrame, sampled_ids: list[str]) -> pd.DataFrame:
	"""Compute required objective rating stats from interactions only."""
	sampled_set = set(sampled_ids)
	df = interactions_df.copy()
	df["book_id"] = df["book_id"].astype("string").str.strip()

	rating_numeric = pd.to_numeric(df["rating"], errors="coerce")
	df = df.loc[df["book_id"].isin(sampled_set) & (rating_numeric > 0), ["book_id"]].copy()
	df["rating"] = rating_numeric.loc[df.index].astype(int)

	invalid_rating = ~df["rating"].isin([1, 2, 3, 4, 5])
	if invalid_rating.any():
		bad_values = sorted(df.loc[invalid_rating, "rating"].unique().tolist())
		raise ValueError(f"Found rating values outside 1-5: {bad_values[:10]}")

	grouped = df.groupby("book_id", sort=False)
	rows = []
	for book_id in sampled_ids:
		if book_id not in grouped.groups:
			raise ValueError(f"Sampled book {book_id} missing from interactions with rating > 0")

		book_df = grouped.get_group(book_id)
		ratings = book_df["rating"].to_numpy(dtype=float)
		counts = (
			book_df["rating"]
			.value_counts()
			.reindex([1, 2, 3, 4, 5], fill_value=0)
			.astype(int)
		)
		count_dict = {str(i): int(counts.loc[i]) for i in [1, 2, 3, 4, 5]}

		rows.append(
			{
				"book_id": book_id,
				"average_rating": float(np.mean(ratings)),
				"rating_std_dev": float(np.std(ratings, ddof=0)),
				"rating_distribution_text": format_rating_distribution_text(count_dict),
			}
		)

	stats_df = pd.DataFrame(rows)
	if len(stats_df) != EXPECTED_BOOK_COUNT or not stats_df["book_id"].is_unique:
		raise ValueError("Objective rating stats validation failed")
	return stats_df


def build_reviews_text_block(reviews_for_book: pd.DataFrame) -> str:
	"""Build REVIEW 1..12 text block from sampled reviews in existing sampled order."""
	lines = []
	for i, text in enumerate(reviews_for_book["review_text"].tolist(), start=1):
		cleaned = "" if text is None else str(text).strip()
		lines.append(f"REVIEW {i}:\n{cleaned}")
	return "\n\n".join(lines).strip()


def build_final_dataframe(
	sampled_reviews_df: pd.DataFrame,
	books_df: pd.DataFrame,
	rating_stats_df: pd.DataFrame,
	sampled_ids: list[str],
) -> pd.DataFrame:
	"""Merge titles, objective stats, and sampled reviews into final LLM input schema."""
	sampled_set = set(sampled_ids)

	books = books_df.copy()
	books["book_id"] = books["book_id"].astype("string").str.strip()
	books = books.loc[books["book_id"].isin(sampled_set), ["book_id", "title"]].copy()
	if not books["book_id"].is_unique:
		raise ValueError("books.parquet must provide one title row per sampled book")

	review_counts = sampled_reviews_df.groupby("book_id").size()
	bad_counts = review_counts[review_counts != REVIEWS_PER_BOOK]
	if not bad_counts.empty:
		example = {str(k): int(v) for k, v in bad_counts.head(10).items()}
		raise ValueError(
			f"Sampled review rows must be exactly {REVIEWS_PER_BOOK} per book: {example}"
		)

	grouped = sampled_reviews_df.groupby("book_id", sort=False)
	review_rows = []
	for book_id in sampled_ids:
		if book_id not in grouped.groups:
			raise ValueError(f"Missing sampled reviews for book {book_id}")
		text_block = build_reviews_text_block(grouped.get_group(book_id))
		review_rows.append({"book_id": book_id, "reviews_text_block": text_block})
	review_blocks = pd.DataFrame(review_rows)

	merged = books.merge(rating_stats_df, on="book_id", how="inner")
	merged = merged.merge(review_blocks, on="book_id", how="inner")

	final_df = merged[
		[
			"book_id",
			"title",
			"average_rating",
			"rating_std_dev",
			"rating_distribution_text",
			"reviews_text_block",
		]
	].copy()

	if len(final_df) != EXPECTED_BOOK_COUNT:
		raise ValueError(
			f"Final output must contain {EXPECTED_BOOK_COUNT} rows, found {len(final_df)}"
		)
	if not final_df["book_id"].is_unique:
		raise ValueError("Final output contains duplicate book_id values")
	if set(final_df["book_id"].astype(str)) != set(sampled_ids):
		raise ValueError("Final output book coverage does not match sampled book list")

	if final_df["reviews_text_block"].astype("string").str.strip().eq("").any():
		raise ValueError("Final output contains empty reviews_text_block values")
	if final_df["rating_distribution_text"].astype("string").str.strip().eq("").any():
		raise ValueError("Final output contains empty rating_distribution_text values")

	return final_df


def build_summary(
	sampled_ids: list[str],
	final_df: pd.DataFrame,
	drop_counts: dict[str, int],
	clean_count_stats: dict[str, float | int],
) -> dict[str, Any]:
	"""Build concise deterministic summary for final controversy prep output."""
	lengths = final_df["reviews_text_block"].map(lambda x: len(str(x)))

	return {
		"sampled_books_expected": len(sampled_ids),
		"final_output_rows": int(len(final_df)),
		"all_book_ids_unique": bool(final_df["book_id"].is_unique),
		"all_books_have_exactly_12_reviews": True,
		"min_word_count_threshold": MIN_WORD_COUNT,
		"language_detector": "langid",
		"random_seed": RANDOM_SEED,
		"drop_counts": {
			"dropped_because_book_id_not_in_sampled_list": drop_counts[
				"dropped_because_book_id_not_in_sampled_list"
			],
			"dropped_because_rating_lte_0": drop_counts["dropped_because_rating_lte_0"],
			"dropped_because_empty_review_text": drop_counts[
				"dropped_because_empty_review_text"
			],
			"dropped_because_duplicate_review_id": drop_counts[
				"dropped_because_duplicate_review_id"
			],
			"dropped_because_duplicate_book_user_review_text": drop_counts[
				"dropped_because_duplicate_book_user_review_text"
			],
			"dropped_because_not_english": drop_counts["dropped_because_not_english"],
			"dropped_because_below_minimum_word_count": drop_counts[
				"dropped_because_below_minimum_word_count"
			],
		},
		"cleaned_review_count_stats": {
			"total_valid_reviews_left": clean_count_stats["total_valid_reviews_left"],
			"median_reviews_per_book_after_cleaning": clean_count_stats[
				"median_reviews_per_book_after_cleaning"
			],
			"min_reviews_for_a_single_book_after_cleaning": clean_count_stats[
				"min_reviews_for_a_single_book_after_cleaning"
			],
			"max_reviews_for_a_single_book_after_cleaning": clean_count_stats[
				"max_reviews_for_a_single_book_after_cleaning"
			],
		},
		"final_review_block_length_stats": {
			"median_reviews_text_block_char_length": float(lengths.median()),
			"max_reviews_text_block_char_length": int(lengths.max()),
		},
	}


def main() -> None:
	"""Run all controversy prep stages and write only final parquet + summary."""
	root = repo_root()

	sampled_ids_path = root / "data" / "artifacts" / "sampled_book_ids.json"
	books_path = root / "data" / "processed" / "books.parquet"
	interactions_path = root / "data" / "processed" / "interactions.parquet"
	reviews_path = root / "data" / "processed" / "reviews.parquet"

	output_dir = root / "data" / "artifacts" / "controversy_prep"
	output_dir.mkdir(parents=True, exist_ok=True)

	output_parquet_path = output_dir / "book_llm_input.parquet"
	output_summary_path = output_dir / "book_llm_input_summary.json"

	sampled_ids = load_sampled_book_ids(sampled_ids_path)

	books_df = load_required_parquet(books_path, {"book_id", "title"})
	interactions_df = load_required_parquet(interactions_path, {"book_id", "rating"})
	reviews_df = load_required_parquet(
		reviews_path, {"review_id", "book_id", "user_id", "rating", "review_text"}
	)

	clean_reviews_df, drop_counts, clean_count_stats = clean_reviews(reviews_df, sampled_ids)
	clean_reviews_df = assign_bucket_labels(clean_reviews_df)

	sampled_reviews_df = sample_reviews_all_books(clean_reviews_df, sampled_ids)
	rating_stats_df = compute_rating_stats(interactions_df, sampled_ids)

	final_df = build_final_dataframe(sampled_reviews_df, books_df, rating_stats_df, sampled_ids)

	summary = build_summary(
		sampled_ids=sampled_ids,
		final_df=final_df,
		drop_counts=drop_counts,
		clean_count_stats=clean_count_stats,
	)

	final_df.to_parquet(output_parquet_path, index=False)
	with output_summary_path.open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	print("Wrote final controversy prep artifacts:")
	print("- data/artifacts/controversy_prep/book_llm_input.parquet")
	print("- data/artifacts/controversy_prep/book_llm_input_summary.json")


if __name__ == "__main__":
	main()
