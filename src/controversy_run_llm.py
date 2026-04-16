"""Full paid LLM inference run for controversy module.

Run from repo root:
    python src/controversy_run_llm.py
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "claude-haiku-4-5-20251001"
TEMPERATURE = 0.2
MAX_TOKENS = 1500
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0

SYSTEM_PROMPT = """You are an expert literary review analyst. Your task is to analyze a sampled set of reader reviews for one book and produce a structured, evidence-grounded summary.

You will be given:
- the book title
- light book-level rating statistics
- a block of sampled reviews

Your job is to read the reviews carefully, identify the strongest evidence-supported praise and criticism, and then synthesize that evidence into a concise overall judgment and a small set of clean top tags.

CORE PRINCIPLES

1. PRIMARY EVIDENCE RULE
Use the review text as the primary evidence.
The title and rating statistics are only supporting context.
You may use the title to understand names, setting, or references mentioned in the reviews, but you must not use outside knowledge to add praise, criticism, plot details, themes, or opinions that are not explicitly supported by the provided reviews.

2. DO NOT FORCE CONTROVERSY OR BALANCE
If the reviews are mostly positive, reflect that honestly.
If the reviews are mostly negative, reflect that honestly.
Do not invent criticism or praise just to create symmetry.
If there is little or no evidence for one side, return an empty array for that field.

3. BE CONSERVATIVE WHEN EVIDENCE IS WEAK
If the reviews are vague, repetitive, or not detailed enough to support specific insights, stay conservative.
Do not hallucinate specificity.
If a small portion of the provided review text is clearly non-English, noisy, or corrupted, ignore that portion and base the summary on the remaining coherent English review evidence.
If evidence is extremely sparse, use this exact sentence for `overall_judgment`:
"The provided reviews lack sufficient detail to form a comprehensive judgment."

4. PRIORITIZE SALIENT AND REPEATED THEMES
Focus on the most central, repeated, or strongly stated review themes.
Do not include minor, speculative, or weakly supported points.

5. INTERNAL EVIDENCE LAYER
`positive_aspects` and `negative_aspects` are not UI-facing fields. They are internal evidence summaries used to support later synthesis.
They may contain:
- short phrases
- aspect labels
- concise evidence-grounded statements
- short sentences
They do not need to be ultra-short or polished.
Their purpose is to preserve meaningful evidence before generating the final summary and tags.

FIELD-BY-FIELD REQUIREMENTS

1. `book_id`
Return the provided book ID exactly.

2. `positive_aspects`
Return an array of the most important positive themes supported by the reviews.
These may be short phrases or short evidence-grounded statements.
They should capture concrete strengths, recurring praise, or clearly positive reader reactions.
They do not need to be uniform in style, but they must remain concise and evidence-based.
If there is no meaningful positive evidence, return [].

3. `negative_aspects`
Return an array of the most important negative themes supported by the reviews.
These may be short phrases or short evidence-grounded statements.
They should capture concrete weaknesses, recurring criticism, or clearly negative reader reactions.
They do not need to be uniform in style, but they must remain concise and evidence-based.
If there is no meaningful negative evidence, return [].

4. `overall_judgment`
Write 2 to 3 sentences, roughly 40 to 80 words.
This is the main UI-facing explanation.
It should synthesize the strongest evidence from the positive and negative aspects.
If readers are split on one issue, describe that explicitly instead of producing contradictory claims.
For example, say that a theme proved divisive or drew mixed reactions.
Do not mention that you were given sampled reviews or rating statistics.

5. `top_tags`
Return 2 to 4 tags that best summarize the strongest overall themes.
These are UI-facing tags.
Rules:
- each tag must be 1 to 4 words
- tags must be STRICTLY UPPERCASE
- tags must be coherent and non-contradictory
- do not produce both positive and negative versions of the same idea
- if the reviews are mixed on one issue, use a synthesized tag like DIVISIVE PACING, MIXED CHARACTER REACTIONS, or POLARIZING TONE
- Always return 2 to 4 tags. If the evidence is weak or highly mixed, use broader but still evidence-grounded tags such as MIXED RECEPTION, CHARACTER-DRIVEN STORY, SLOW PACING, FAMILY DRAMA, HISTORICAL CONTEXT, DARK TONE, or UNEVEN EXECUTION. Do not return an empty top_tags array.

OUTPUT STYLE RULES

- Be faithful to the reviews.
- Prefer grounded synthesis over completeness.
- Do not overstate weak evidence.
- Do not use outside knowledge to fill gaps.
- The final answer must be consistent: `overall_judgment` and `top_tags` should be supported by the evidence captured in `positive_aspects` and `negative_aspects`.

Return only the structured output in the required schema."""

TOOLS = [
    {
        "type": "custom",
        "name": "generate_review_summary",
        "description": "Generates a structured, evidence-grounded summary of book reviews.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "book_id": {
                    "type": "string",
                    "description": "Return the exact book ID from the input with no modification.",
                },
                "positive_aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 6,
                    "description": (
                        "Important positive themes, strengths, or short evidence-grounded "
                        "statements supported by the reviews. May contain phrases or short "
                        "sentences. Return [] if no meaningful positive evidence exists."
                    ),
                },
                "negative_aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": 6,
                    "description": (
                        "Important negative themes, weaknesses, or short evidence-grounded "
                        "statements supported by the reviews. May contain phrases or short "
                        "sentences. Return [] if no meaningful negative evidence exists."
                    ),
                },
                "overall_judgment": {
                    "type": "string",
                    "description": (
                        "A concise 2 to 3 sentence synthesis, about 40 to 80 words, suitable "
                        "for direct UI display. It should summarize the strongest supported "
                        "praise and criticism without hallucinating balance."
                    ),
                },
                "top_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4,
                    "description": (
                        "Two to four short summary tags. Each tag should be 1 to 4 words, "
                        "strictly uppercase, coherent, and non-contradictory. Use synthesized "
                        "tags like DIVISIVE PACING when reactions are mixed."
                    ),
                },
            },
            "required": [
                "book_id",
                "positive_aspects",
                "negative_aspects",
                "overall_judgment",
                "top_tags",
            ],
        },
    }
]


def repo_root() -> Path:
    """Return repository root based on this file location."""
    return Path(__file__).resolve().parent.parent


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def json_dumps_compact(obj: Any) -> str:
    """Serialize JSON with stable compact formatting."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_input_parquet(input_path: Path) -> pd.DataFrame:
    """Load and validate deterministic LLM input parquet."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_path}")

    df = pd.read_parquet(input_path)
    required_columns = {
        "book_id",
        "title",
        "average_rating",
        "rating_std_dev",
        "rating_distribution_text",
        "reviews_text_block",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Input parquet missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["book_id"] = df["book_id"].astype("string").str.strip()
    if df["book_id"].isna().any() or (df["book_id"] == "").any():
        raise ValueError("Input parquet has empty book_id values.")

    duplicate_ids = df[df["book_id"].duplicated()]["book_id"].tolist()
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Input parquet has duplicate book_id rows (examples: {preview}).")

    return df.reset_index(drop=True)


def iter_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records; skip malformed lines gracefully."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSONL line {idx} in {path.name}")
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def load_existing_ids(jsonl_path: Path) -> set[str]:
    """Load unique non-empty book IDs from a JSONL artifact file."""
    ids: set[str] = set()
    for record in iter_jsonl_records(jsonl_path):
        book_id = str(record.get("book_id", "")).strip()
        if book_id:
            ids.add(book_id)
    return ids


def build_user_message(row: pd.Series) -> str:
    """Build the exact per-book user message block."""
    return (
        "<book_info>\n"
        f"  <book_id>{row['book_id']}</book_id>\n"
        f"  <title>{row['title']}</title>\n"
        f"  <average_rating>{row['average_rating']}</average_rating>\n"
        f"  <rating_std_dev>{row['rating_std_dev']}</rating_std_dev>\n"
        f"  <rating_distribution>{row['rating_distribution_text']}</rating_distribution>\n"
        "</book_info>\n\n"
        "<reviews>\n"
        f"{row['reviews_text_block']}\n"
        "</reviews>"
    )


def call_claude_for_book(client: anthropic.Anthropic, user_message: str) -> Any:
    """Call Claude once for a single book."""
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [{"type": "text", "text": user_message}]}],
        tools=TOOLS,
        tool_choice={"type": "tool", "name": "generate_review_summary"},
        thinking={"type": "disabled"},
    )


class ClaudeCallError(Exception):
    """Wrap Claude call failures with retry-attempt metadata."""

    def __init__(self, message: str, attempts: int, original_error: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.original_error = original_error


def is_transient_api_error(exc: Exception) -> bool:
    """Return True if an exception is likely transient and retryable."""
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "APIStatusError",
    }
    if exc.__class__.__name__ in transient_names:
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def call_claude_with_retry(
    client: anthropic.Anthropic,
    user_message: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[Any, int]:
    """Call Claude with retry/backoff on transient API failures."""
    attempt = 0
    while True:
        attempt += 1
        try:
            response = call_claude_for_book(client, user_message=user_message)
            return response, attempt
        except Exception as exc:
            retryable = is_transient_api_error(exc)
            if not retryable or attempt >= max_retries:
                raise ClaudeCallError(
                    message=str(exc),
                    attempts=attempt,
                    original_error=exc,
                ) from exc
            sleep_seconds = min(30.0, BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            sleep_seconds += random.uniform(0.0, 0.5)
            print(
                f"  transient API error on attempt {attempt}/{max_retries}: "
                f"{exc.__class__.__name__}; retrying in {sleep_seconds:.2f}s"
            )
            time.sleep(sleep_seconds)


def extract_tool_payload(response: Any) -> dict[str, Any]:
    """Extract generate_review_summary tool payload from API response."""
    for block in getattr(response, "content", []):
        block_type = getattr(block, "type", None)
        block_name = getattr(block, "name", None)
        if block_type == "tool_use" and block_name == "generate_review_summary":
            payload = getattr(block, "input", None)
            if not isinstance(payload, dict):
                raise ValueError("Tool-use payload missing or invalid.")
            return payload

        if isinstance(block, dict):
            if block.get("type") == "tool_use" and block.get("name") == "generate_review_summary":
                payload = block.get("input")
                if not isinstance(payload, dict):
                    raise ValueError("Tool-use payload missing or invalid.")
                return payload
    raise ValueError("No generate_review_summary tool-use payload found in response.")


def _coerce_string_list(value: Any) -> list[str]:
    """Coerce mixed model output into cleaned string list."""
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def normalize_structured_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize output and clamp list lengths to schema maximums."""
    normalized = dict(payload)
    normalized["book_id"] = str(normalized.get("book_id", "")).strip()
    normalized["positive_aspects"] = _coerce_string_list(normalized.get("positive_aspects"))[:6]
    normalized["negative_aspects"] = _coerce_string_list(normalized.get("negative_aspects"))[:6]
    normalized["top_tags"] = _coerce_string_list(normalized.get("top_tags"))[:4]
    normalized["overall_judgment"] = str(normalized.get("overall_judgment", "")).strip()
    return normalized


def validate_structured_output(payload: dict[str, Any], input_book_id: str) -> None:
    """Validate required payload fields and semantic constraints."""
    required = [
        "book_id",
        "positive_aspects",
        "negative_aspects",
        "overall_judgment",
        "top_tags",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if payload["book_id"] != input_book_id:
        raise ValueError(
            f"Returned book_id mismatch. Expected {input_book_id}, got {payload['book_id']}"
        )

    if not isinstance(payload["positive_aspects"], list):
        raise ValueError("positive_aspects must be a list")
    if not isinstance(payload["negative_aspects"], list):
        raise ValueError("negative_aspects must be a list")
    if not isinstance(payload["top_tags"], list):
        raise ValueError("top_tags must be a list")

    tag_count = len(payload["top_tags"])
    if tag_count < 2 or tag_count > 4:
        raise ValueError(f"top_tags length must be between 2 and 4; got {tag_count}")

    if not payload["overall_judgment"]:
        raise ValueError("overall_judgment must be a non-empty string")


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json_dumps_compact(record) + "\n")


def rebuild_output_parquet(success_jsonl_path: Path, output_parquet_path: Path) -> int:
    """Rebuild final parquet from append-only success JSONL."""
    records = iter_jsonl_records(success_jsonl_path)
    if not records:
        pd.DataFrame(
            columns=[
                "book_id",
                "title",
                "average_rating",
                "rating_std_dev",
                "rating_distribution_text",
                "input_reviews_char_length",
                "model_name",
                "temperature",
                "max_tokens",
                "api_response_id",
                "stop_reason",
                "usage_output_tokens",
                "positive_aspects",
                "negative_aspects",
                "overall_judgment",
                "top_tags",
                "timestamp_utc",
            ]
        ).to_parquet(output_parquet_path, index=False)
        return 0

    rows: list[dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "book_id": str(rec.get("book_id", "")),
                "title": rec.get("title"),
                "average_rating": rec.get("average_rating"),
                "rating_std_dev": rec.get("rating_std_dev"),
                "rating_distribution_text": rec.get("rating_distribution_text"),
                "input_reviews_char_length": rec.get("input_reviews_char_length"),
                "model_name": rec.get("model_name"),
                "temperature": rec.get("temperature"),
                "max_tokens": rec.get("max_tokens"),
                "api_response_id": rec.get("api_response_id"),
                "stop_reason": rec.get("stop_reason"),
                "usage_output_tokens": rec.get("usage_output_tokens"),
                "positive_aspects": json_dumps_compact(rec.get("positive_aspects", [])),
                "negative_aspects": json_dumps_compact(rec.get("negative_aspects", [])),
                "overall_judgment": rec.get("overall_judgment"),
                "top_tags": json_dumps_compact(rec.get("top_tags", [])),
                "timestamp_utc": rec.get("timestamp_utc"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["book_id"], keep="last").reset_index(drop=True)
    df.to_parquet(output_parquet_path, index=False)
    return len(df)


def build_run_summary(
    *,
    input_books_expected: int,
    already_completed_before_this_run: int,
    books_attempted_this_run: int,
    books_succeeded_this_run: int,
    books_failed_this_run: int,
    books_completed_total: int,
    books_failed_total: int,
    books_remaining: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    avg_input_chars_success: float | None,
    avg_output_tokens_success: float | None,
    run_started_utc: str,
    run_finished_utc: str,
) -> dict[str, Any]:
    """Build concise run metadata summary."""
    return {
        "input_books_expected": input_books_expected,
        "already_completed_before_this_run": already_completed_before_this_run,
        "books_attempted_this_run": books_attempted_this_run,
        "books_succeeded_this_run": books_succeeded_this_run,
        "books_failed_this_run": books_failed_this_run,
        "books_completed_total": books_completed_total,
        "books_failed_total": books_failed_total,
        "books_remaining": books_remaining,
        "all_input_books_processed": books_remaining == 0,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "resume_mode": True,
        "average_input_review_block_character_length_for_successes": avg_input_chars_success,
        "average_output_token_count_for_successes": avg_output_tokens_success,
        "run_started_utc": run_started_utc,
        "run_finished_utc": run_finished_utc,
    }


def main() -> None:
    """Run full controversy LLM inference with checkpointing and resume safety."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing ANTHROPIC_API_KEY environment variable. "
            "Add it to your environment or .env before running."
        )

    run_started_utc = utc_now_iso()
    root = repo_root()
    input_path = root / "data" / "artifacts" / "controversy_prep" / "book_llm_input.parquet"
    output_dir = root / "data" / "artifacts" / "controversy_run"
    success_jsonl_path = output_dir / "book_llm_outputs.jsonl"
    failure_jsonl_path = output_dir / "book_llm_failures.jsonl"
    summary_path = output_dir / "book_llm_run_summary.json"
    output_parquet_path = output_dir / "book_llm_outputs.parquet"

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_input_parquet(input_path)

    existing_success_ids = load_existing_ids(success_jsonl_path)
    existing_failure_ids = load_existing_ids(failure_jsonl_path)

    pending_df = df[~df["book_id"].isin(existing_success_ids)].copy().reset_index(drop=True)

    total_expected = len(df)
    already_completed = len(existing_success_ids)
    total_to_process = len(pending_df)
    print(
        f"Loaded {total_expected} books. "
        f"Already completed: {already_completed}. "
        f"Processing now: {total_to_process}."
    )

    client = anthropic.Anthropic(api_key=api_key)

    attempted_this_run = 0
    succeeded_this_run = 0
    failed_this_run = 0
    success_input_char_lengths: list[int] = []
    success_output_tokens: list[int] = []

    for idx, (_, row) in enumerate(pending_df.iterrows(), start=1):
        book_id = str(row["book_id"])
        title = str(row["title"])
        reviews_text = str(row["reviews_text_block"])
        input_len = len(reviews_text)

        print(f"[{idx}/{total_to_process}] book_id={book_id} | title={title}")
        attempted_this_run += 1

        try:
            user_message = build_user_message(row)
            response, attempt_count = call_claude_with_retry(
                client=client,
                user_message=user_message,
                max_retries=MAX_RETRIES,
            )
            payload = extract_tool_payload(response)
            payload = normalize_structured_output(payload)
            validate_structured_output(payload, input_book_id=book_id)

            usage = getattr(response, "usage", None)
            output_tokens = getattr(usage, "output_tokens", None) if usage else None
            output_tokens_int = output_tokens if isinstance(output_tokens, int) else None
            if output_tokens_int is not None:
                success_output_tokens.append(output_tokens_int)
            success_input_char_lengths.append(input_len)

            success_record = {
                "book_id": book_id,
                "title": title,
                "average_rating": float(row["average_rating"]),
                "rating_std_dev": float(row["rating_std_dev"]),
                "rating_distribution_text": str(row["rating_distribution_text"]),
                "input_reviews_char_length": input_len,
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "api_response_id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "usage_output_tokens": output_tokens_int,
                "positive_aspects": payload["positive_aspects"],
                "negative_aspects": payload["negative_aspects"],
                "overall_judgment": payload["overall_judgment"],
                "top_tags": payload["top_tags"],
                "timestamp_utc": utc_now_iso(),
            }
            append_jsonl_record(success_jsonl_path, success_record)
            existing_success_ids.add(book_id)
            succeeded_this_run += 1
            print(f"  success (attempts={attempt_count})")

        except Exception as exc:
            failed_this_run += 1
            attempt_count = 1
            error_for_logging = exc
            if isinstance(exc, ClaudeCallError):
                attempt_count = exc.attempts
                error_for_logging = exc.original_error
            error_type = error_for_logging.__class__.__name__
            failure_record = {
                "book_id": book_id,
                "title": title,
                "error_type": error_type,
                "error_message": str(error_for_logging),
                "attempt_count": attempt_count,
                "timestamp_utc": utc_now_iso(),
            }
            append_jsonl_record(failure_jsonl_path, failure_record)
            existing_failure_ids.add(book_id)
            print(f"  failure ({error_type}, attempts={attempt_count}): {error_for_logging}")
            continue

    success_ids_after = load_existing_ids(success_jsonl_path)
    failure_ids_after = load_existing_ids(failure_jsonl_path)
    input_book_ids = set(df["book_id"].astype(str).tolist())
    completed_total = len(success_ids_after)
    unresolved_failed_total = len(failure_ids_after.difference(success_ids_after))

    parquet_rows = rebuild_output_parquet(success_jsonl_path, output_parquet_path)
    run_finished_utc = utc_now_iso()

    processed_ids = success_ids_after.union(failure_ids_after).intersection(input_book_ids)
    books_remaining = total_expected - len(processed_ids)
    books_remaining = max(0, books_remaining)

    avg_input_chars_success = (
        float(sum(success_input_char_lengths) / len(success_input_char_lengths))
        if success_input_char_lengths
        else None
    )
    avg_output_tokens_success = (
        float(sum(success_output_tokens) / len(success_output_tokens))
        if success_output_tokens
        else None
    )

    summary = build_run_summary(
        input_books_expected=total_expected,
        already_completed_before_this_run=already_completed,
        books_attempted_this_run=attempted_this_run,
        books_succeeded_this_run=succeeded_this_run,
        books_failed_this_run=failed_this_run,
        books_completed_total=completed_total,
        books_failed_total=unresolved_failed_total,
        books_remaining=books_remaining,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        max_retries=MAX_RETRIES,
        avg_input_chars_success=avg_input_chars_success,
        avg_output_tokens_success=avg_output_tokens_success,
        run_started_utc=run_started_utc,
        run_finished_utc=run_finished_utc,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nRun complete.")
    print(f"Successful books in parquet: {parquet_rows}")
    print(f"Wrote: {success_jsonl_path.name}, {failure_jsonl_path.name}, {summary_path.name}, {output_parquet_path.name}")


if __name__ == "__main__":
    main()
