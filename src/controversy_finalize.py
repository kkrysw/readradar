"""Create final clean controversy artifacts for sharing.

Run from repo root:
    python src/controversy_finalize.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

FINAL_COLUMNS = [
    "book_id",
    "title",
    "average_rating",
    "rating_std_dev",
    "overall_judgment",
    "top_tags",
]
EXPECTED_SAMPLED_COUNT = 5000


def repo_root() -> Path:
    """Return repository root based on this file location."""
    return Path(__file__).resolve().parent.parent


def json_dumps_compact(obj: Any) -> str:
    """Serialize compact JSON for stable parquet string values."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_sampled_book_ids(path: Path) -> list[str]:
    """Load sampled IDs and validate exactly 5000 unique IDs."""
    if not path.exists():
        raise FileNotFoundError(f"Missing sampled book list: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("sampled_book_ids.json must be a JSON array of book IDs.")

    sampled_ids = [str(x).strip() for x in data]
    if any(not bid for bid in sampled_ids):
        raise ValueError("sampled_book_ids.json contains empty book IDs.")

    unique_ids = set(sampled_ids)
    if len(unique_ids) != EXPECTED_SAMPLED_COUNT:
        raise ValueError(
            f"sampled_book_ids.json must contain exactly {EXPECTED_SAMPLED_COUNT} unique book IDs; "
            f"found {len(unique_ids)}."
        )
    if len(sampled_ids) != EXPECTED_SAMPLED_COUNT:
        raise ValueError(
            f"sampled_book_ids.json must contain exactly {EXPECTED_SAMPLED_COUNT} rows; "
            f"found {len(sampled_ids)}."
        )

    return sampled_ids


def load_run_outputs(path: Path) -> pd.DataFrame:
    """Load run parquet and ensure required columns and unique IDs."""
    if not path.exists():
        raise FileNotFoundError(f"Missing run parquet: {path}")

    df = pd.read_parquet(path).copy()
    required_run_columns = set(FINAL_COLUMNS)
    missing = sorted(required_run_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Run parquet missing required columns: {', '.join(missing)}")

    df["book_id"] = df["book_id"].astype("string").str.strip()
    if df["book_id"].isna().any() or (df["book_id"] == "").any():
        raise ValueError("Run parquet contains empty book_id values.")

    if df["book_id"].duplicated().any():
        dup_examples = df.loc[df["book_id"].duplicated(), "book_id"].head(5).tolist()
        raise ValueError(f"Run parquet contains duplicate book_ids (examples: {dup_examples}).")

    return df


def normalize_top_tags(value: Any) -> tuple[Any, str]:
    """Normalize top_tags to a consistent non-empty storage format."""
    if isinstance(value, list):
        tags = [str(x).strip() for x in value if str(x).strip()]
        return tags, "list"

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "", "json_string"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Keep plain text fallback as string if non-empty.
            return stripped, "json_string"
        if isinstance(parsed, list):
            tags = [str(x).strip() for x in parsed if str(x).strip()]
            return json_dumps_compact(tags), "json_string"
        return stripped, "json_string"

    if value is None:
        return "", "json_string"
    return str(value).strip(), "json_string"


def validate_final_dataframe(final_df: pd.DataFrame, sampled_ids: list[str]) -> dict[str, Any]:
    """Validate final dataframe against required schema and sampled IDs."""
    missing_cols = [col for col in FINAL_COLUMNS if col not in final_df.columns]
    if missing_cols:
        raise ValueError(f"Final dataframe missing required columns: {missing_cols}")

    final_ids = final_df["book_id"].astype(str).str.strip().tolist()
    final_id_set = set(final_ids)
    sampled_id_set = set(sampled_ids)

    if len(final_df) != EXPECTED_SAMPLED_COUNT:
        raise ValueError(
            f"Final output row count must be exactly {EXPECTED_SAMPLED_COUNT}; found {len(final_df)}."
        )

    if len(final_id_set) != len(final_ids):
        raise ValueError("Final output contains duplicate book_ids.")

    missing_from_final = sampled_id_set.difference(final_id_set)
    extra_in_final = final_id_set.difference(sampled_id_set)
    if missing_from_final or extra_in_final:
        raise ValueError(
            "Final book_id set mismatch: "
            f"missing_from_final={len(missing_from_final)}, extra_in_final={len(extra_in_final)}."
        )

    overall_non_empty = final_df["overall_judgment"].astype(str).str.strip().ne("").all()
    if not overall_non_empty:
        raise ValueError("overall_judgment must be non-empty for every row.")

    def _top_tags_non_empty(v: Any) -> bool:
        if isinstance(v, list):
            return any(str(x).strip() for x in v)
        return bool(str(v).strip())

    top_tags_non_empty = final_df["top_tags"].apply(_top_tags_non_empty).all()
    if not top_tags_non_empty:
        raise ValueError("top_tags must be non-empty for every row.")

    return {
        "missing_from_final_count": len(missing_from_final),
        "extra_in_final_count": len(extra_in_final),
        "all_overall_judgment_non_empty": bool(overall_non_empty),
        "all_top_tags_non_empty": bool(top_tags_non_empty),
        "book_id_sets_match": True,
        "all_book_ids_unique": True,
    }


def build_summary(
    *,
    final_df: pd.DataFrame,
    top_tags_storage_format: str,
    validation_details: dict[str, Any],
) -> dict[str, Any]:
    """Build concise finalization summary."""
    return {
        "sampled_books_expected": EXPECTED_SAMPLED_COUNT,
        "final_output_rows": int(len(final_df)),
        "all_book_ids_unique": validation_details["all_book_ids_unique"],
        "book_id_sets_match": validation_details["book_id_sets_match"],
        "missing_from_final": validation_details["missing_from_final_count"],
        "extra_in_final": validation_details["extra_in_final_count"],
        "columns_written": FINAL_COLUMNS,
        "top_tags_storage_format": top_tags_storage_format,
        "all_overall_judgment_non_empty": validation_details["all_overall_judgment_non_empty"],
        "all_top_tags_non_empty": validation_details["all_top_tags_non_empty"],
    }


def main() -> None:
    """Finalize controversy artifact from run parquet into clean shareable outputs."""
    root = repo_root()
    sampled_ids_path = root / "data" / "artifacts" / "sampled_book_ids.json"
    run_parquet_path = root / "data" / "artifacts" / "controversy_run" / "book_llm_outputs.parquet"
    final_parquet_path = root / "data" / "artifacts" / "controversy_final.parquet"
    final_summary_path = root / "data" / "artifacts" / "controversy_final_summary.json"

    sampled_ids = load_sampled_book_ids(sampled_ids_path)
    run_df = load_run_outputs(run_parquet_path)

    final_df = run_df[FINAL_COLUMNS].copy()
    final_df["book_id"] = final_df["book_id"].astype(str).str.strip()
    final_df["overall_judgment"] = final_df["overall_judgment"].astype(str).str.strip()

    normalized_top_tags: list[Any] = []
    storage_formats: set[str] = set()
    for value in final_df["top_tags"].tolist():
        normalized, storage_format = normalize_top_tags(value)
        normalized_top_tags.append(normalized)
        storage_formats.add(storage_format)

    # Use JSON string storage if any row cannot be cleanly kept as list.
    top_tags_storage_format = "list" if storage_formats == {"list"} else "json_string"
    if top_tags_storage_format == "json_string":
        final_df["top_tags"] = [
            tags if isinstance(tags, str) else json_dumps_compact(tags) for tags in normalized_top_tags
        ]
    else:
        final_df["top_tags"] = normalized_top_tags

    validation_details = validate_final_dataframe(final_df, sampled_ids)

    final_df = final_df[FINAL_COLUMNS]
    final_df.to_parquet(final_parquet_path, index=False)

    summary = build_summary(
        final_df=final_df,
        top_tags_storage_format=top_tags_storage_format,
        validation_details=validation_details,
    )
    with final_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote controversy final artifacts.")
    print(f"rows={len(final_df)}, top_tags_storage_format={top_tags_storage_format}")


if __name__ == "__main__":
    main()
