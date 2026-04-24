"""
build_book_clusters.py — Reading Neighborhoods artifact builder.

Runs spherical k-means++ (implemented from scratch in `src/clustering.py`)
over the L2-normalized search embeddings, labels each cluster from the
books' noisy Goodreads `popular_shelves` strings, picks representative
books per cluster, and writes two artifacts the Streamlit app consumes.

Inputs:
    data/artifacts/search_embeddings.npy
    data/artifacts/search_books.parquet

Outputs:
    data/artifacts/book_clusters.parquet
    data/artifacts/book_cluster_summary.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Let `scripts/build_book_clusters.py` import from `src/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import (
    rank_cluster_representatives,
    spherical_kmeans,
)


# ─── paths + constants ────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("data/artifacts")

IN_EMBEDDINGS = ARTIFACTS_DIR / "search_embeddings.npy"
IN_BOOKS      = ARTIFACTS_DIR / "search_books.parquet"

OUT_CLUSTERS  = ARTIFACTS_DIR / "book_clusters.parquet"
OUT_SUMMARY   = ARTIFACTS_DIR / "book_cluster_summary.json"

N_CLUSTERS    = 24
MAX_ITER      = 100
TOL           = 1e-5
RANDOM_SEED   = 42
EXPECTED_BOOK_COUNT = 5_000
REPRESENTATIVES_PER_CLUSTER = 5


# ─── shelf cleaning ───────────────────────────────────────────────────────
# Generic / non-semantic shelves — always dropped before label counting.
GENERIC_SHELVES = frozenset({
    "to-read", "currently-reading", "read", "owned", "default",
    "favorites", "favorite", "favourite", "favourites", "favorite-books",
    "favorites-owned", "all-time-favorites",
    "books-i-own", "book-i-own", "books-i-have", "owned-books",
    "own-it", "i-own", "mine", "have", "need", "want",
    "want-to-read", "need-to-buy", "need-to-get", "to-buy",
    "wish-list", "wishlist", "maybe",
    "my-books", "my-library", "my-shelf", "home-library",
    "library", "library-books", "library-book", "borrowed",
    "kindle", "ebook", "ebooks", "e-book", "e-books", "nook",
    "pdf", "calibre", "audiobook", "audiobooks", "audio",
    "audio-book", "audio-books", "audible",
    "paperback", "books",
    "book-club", "bookclub",
    "classroom", "classroom-library", "school",
    "reviewed", "finished",
    "completed-series", "series-to-read", "series-to-start",
    "part-of-a-series", "first-in-series", "first-in-a-series",
    "1st-in-series", "1st-in-a-series",
    "dnf", "did-not-finish", "didn-t-finish", "couldn-t-finish",
    "abandoned", "unfinished", "never-finished", "not-finished",
    "stopped-reading", "gave-up-on", "dropped", "paused", "on-hold",
    "half-read", "not-interested", "nope", "never-again",
    "must-read", "other",
})

# Small explicit series/author blocklist (from the dataset samples).
AUTHOR_SERIES_BLOCKLIST = frozenset({
    "lynsay-sands", "lyndsay-sands", "sands", "sands-lynsay",
    "mari-mancusi", "john-connolly",
    "argeneau", "argeneau-series",
    "argeneau-vampires", "argeneau-vampire-series",
    "blood-coven", "blood-coven-series",
    "charlie-parker-series",
})

# Common variant normalizations. The goal is to collapse near-duplicate tokens
# *before* counting so label selection never has to de-duplicate after the fact.
NORMALIZATION_MAP: dict[str, str] = {
    # science fiction family
    "sci-fi": "science-fiction", "scifi": "science-fiction", "sf": "science-fiction",
    "sci-fi-fantasy": "science-fiction", "scifi-fantasy": "science-fiction",
    "science-fiction-fantasy": "science-fiction",
    "speculative-fiction": "science-fiction", "speculative": "science-fiction",

    # graphic novels / comics family
    "graphic": "graphic-novels",
    "graphic-novel": "graphic-novels",
    "comic": "graphic-novels", "comics": "graphic-novels",
    "comic-book": "graphic-novels", "comic-books": "graphic-novels",
    "comics-graphic-novels": "graphic-novels",
    "graphic-novels-comics": "graphic-novels",
    "comics-and-graphic-novels": "graphic-novels",
    "graphic-novels-and-comics": "graphic-novels",

    # young adult
    "ya": "young-adult", "y-a": "young-adult", "ya-lit": "young-adult",
    "ya-books": "young-adult", "ya-fiction": "young-adult",
    "young": "young-adult", "young-adult-fiction": "young-adult",

    # children
    "children-s": "children", "childrens": "children",
    "childrens-books": "children", "children-s-books": "children",
    "children-s-literature": "children", "childrens-lit": "children",
    "children-s-lit": "children", "children-s-fiction": "children",
    "childrens-fiction": "children", "kid-lit": "children",
    "kids": "children", "kids-books": "children",
    "j-fiction": "juvenile-fiction",
    "middle-grades": "middle-grade",

    # historical — as a Goodreads shelf this almost always tags historical fiction
    "historical": "historical-fiction",
    "historical_fiction": "historical-fiction",
    "historical-fic": "historical-fiction",

    # mystery / crime / thriller composites → canonical single form
    "mystery-thriller": "thriller", "mystery-thrillers": "thriller",
    "thrillers": "thriller",
    "crime-fiction": "crime",
    "detective-fiction": "detective",
    "suspense-thriller": "suspense",

    # religion / spirituality
    "religious": "religion", "religious-fiction": "religion",
    "spiritual": "spirituality",
    "christianity": "christian",

    # nonfiction / memoir / biography
    "non-fiction": "nonfiction",
    "bio": "biography", "biographies": "biography", "biographical": "biography",
    "autobiographies": "autobiography",
    "memoirs": "memoir",

    # broad composites → collapse to the broad parent so downweighting applies
    "adult-fiction": "fiction",
    "general-fiction": "fiction",
    "classic": "classics",

    # bare place/region hints → usable literary-region label when present
    "england": "british-literature",
    "british": "british-literature",
    "uk": "british-literature",

    # war
    "wwii": "world-war-ii", "ww2": "world-war-ii", "world-war-2": "world-war-ii",

    # vampires / paranormal romance
    "vampire": "vampires", "vamps": "vampires", "vamp": "vampires",
    "vamp-books": "vampires", "vampire-books": "vampires",
    "vampire-novels": "vampires",
    "pnr": "paranormal-romance", "para-romance": "paranormal-romance",
    "romance-paranormal": "paranormal-romance",
    "adult-romance": "romance", "contemporary-romance": "romance",
    "romantic": "romance",

    # humor / newbery
    "humour": "humor", "humorous": "humor",
    "newberry": "newbery", "newberry-honor": "newbery-honor",
    "newberry-books": "newbery", "newberys": "newbery",
}

# Matches "2005", "read-2010", "read-in-2011", "2011-reads", "read-in-2014", etc.
_YEAR_TAG_RE = re.compile(
    r"""^(?:
        \d{4}             # bare year
        | read-\d{4}      # read-2010
        | read-in-\d{4}   # read-in-2011
        | \d{4}-reads?    # 2011-reads
        | read-\d{2}      # read-11
    )$""",
    re.VERBOSE,
)

# Rating shelves: 1-star, 2-stars, ..., plus "1-stars", etc.
_RATING_SHELF_RE = re.compile(r"^[1-5]-stars?$")


def _normalize_token(tok: str) -> str:
    """Lowercase, strip, convert underscores to hyphens."""
    return tok.strip().lower().replace("_", "-")


def _alpha_count(tok: str) -> int:
    return sum(1 for ch in tok if ch.isalpha() and ch.isascii())


def _is_ascii_alpha_enough(tok: str) -> bool:
    """Require at least 2 ASCII-alphabetic characters; rejects non-Latin scripts."""
    return _alpha_count(tok) >= 2


def clean_shelves(raw: str) -> list[str]:
    """
    Parse a `popular_shelves` string and return the list of cleaned,
    normalized, meaningful shelf tokens. Order is preserved; duplicates
    (after normalization) are kept so the caller can count frequencies
    across books within a cluster.

    Filter chain (applied in order):
        * split on comma, strip, lowercase, underscore -> hyphen
        * drop empty / too-long (> 35 chars)
        * drop tokens with fewer than 2 ASCII-alphabetic chars
          (also filters purely numeric tokens and non-Latin scripts)
        * drop year / read-year tags
        * drop rating shelves (1-star, 2-stars, ...)
        * drop generic/utility shelves (to-read, owned, audiobook, dnf, ...)
        * drop author/series blocklist
        * apply normalization map (sci-fi -> science-fiction, ya -> young-adult, ...)
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    cleaned: list[str] = []
    for piece in raw.split(","):
        tok = _normalize_token(piece)
        if not tok or len(tok) > 35:
            continue
        if not _is_ascii_alpha_enough(tok):
            continue
        if _YEAR_TAG_RE.match(tok) or _RATING_SHELF_RE.match(tok):
            continue
        # Composite utility shelves: "to-read-non-fiction", "currently-reading-ya", ...
        if tok.startswith(("to-read-", "currently-reading-", "already-read-", "re-read")):
            continue
        if tok in GENERIC_SHELVES or tok in AUTHOR_SERIES_BLOCKLIST:
            continue
        # Normalize variants (do this AFTER the drop-lists so originals with
        # distinct spelling like "newberry" still get caught by the map).
        tok = NORMALIZATION_MAP.get(tok, tok)
        if tok in GENERIC_SHELVES or tok in AUTHOR_SERIES_BLOCKLIST:
            continue
        cleaned.append(tok)
    return cleaned


def _title_case_shelf(tok: str) -> str:
    return " ".join(w.capitalize() for w in tok.split("-") if w)


# ─── shelf-count weighting ────────────────────────────────────────────────
# Cluster-central books contribute more to the label decision, so labels
# track the cluster's representatives rather than whatever tag is numerically
# common across the tail.
REP_WEIGHT_TIERS: tuple[tuple[int, float], ...] = ((10, 3.0), (50, 1.5))
BASE_WEIGHT: float = 1.0


def _weight_for_rank(rank: int) -> float:
    for threshold, weight in REP_WEIGHT_TIERS:
        if rank <= threshold:
            return weight
    return BASE_WEIGHT


def weighted_shelf_counts(
    cleaned_shelves: list[list[str]],
    member_indices: np.ndarray,
    cluster_ranks: np.ndarray,
) -> Counter:
    """Representative-weighted shelf frequencies for one cluster."""
    counts: Counter = Counter()
    for idx in member_indices:
        weight = _weight_for_rank(int(cluster_ranks[idx]))
        for shelf in cleaned_shelves[idx]:
            counts[shelf] += weight
    return counts


# ─── display-label resolution ─────────────────────────────────────────────
# Related shelves collapse into a single canonical combined phrase. Both the
# crime-thriller axis and the paranormal axis read better as one phrase per
# family than as three near-synonym tokens in sequence.
FAMILY_MAP: dict[str, str] = {
    # crime-thriller axis
    "mystery":            "Mystery & Thriller",
    "thriller":           "Mystery & Thriller",
    "suspense":           "Mystery & Thriller",
    # crime-detective axis — kept separate from thriller so a Poirot-type
    # cluster can show both axes distinctly.
    "crime":              "Crime & Detective",
    "detective":          "Crime & Detective",
    "noir":               "Crime & Detective",
    # paranormal axis (absorbs "paranormal-romance" on purpose; the Romance
    # shelf still surfaces separately when present)
    "vampires":           "Vampires & Paranormal",
    "paranormal":         "Vampires & Paranormal",
    "supernatural":       "Vampires & Paranormal",
    "paranormal-romance": "Vampires & Paranormal",
    # religion axis
    "religion":           "Religion & Spirituality",
    "spirituality":       "Religion & Spirituality",
    "faith":              "Religion & Spirituality",
    # memoir/bio axis
    "biography":          "Memoir & Biography",
    "memoir":             "Memoir & Biography",
    "autobiography":      "Memoir & Biography",
    # war/history (fiction or nonfiction) axis
    "war":                "War & History",
    "world-war-ii":       "War & History",
    "military":           "War & History",
    # ideas axis (philosophy / psychology; note `science` stays separate so
    # we don't swallow hard-SF or pop-science clusters here)
    "philosophy":         "Philosophy & Ideas",
    "psychology":         "Philosophy & Ideas",
    # self-help axis
    "self-help":          "Self Help",
    "self-improvement":   "Self Help",
    # dystopia axis
    "dystopia":           "Dystopian Fiction",
    "dystopian":          "Dystopian Fiction",
    "post-apocalyptic":   "Dystopian Fiction",
}

# Explicit casing / phrasing overrides for shelves that display on their own
# (i.e. are NOT absorbed into a combined family above). Anything not listed
# here and not in FAMILY_MAP falls through to default title-casing.
DIRECT_DISPLAY: dict[str, str] = {
    # core genres that should keep their own box
    "historical-fiction":    "Historical Fiction",
    "literary-fiction":      "Literary Fiction",
    "science-fiction":       "Science Fiction",
    "classics":              "Classics",
    "fantasy":               "Fantasy",
    "urban-fantasy":         "Urban Fantasy",
    "romance":               "Romance",
    "humor":                 "Humor",
    "horror":                "Horror",
    "graphic-novels":        "Graphic Novels",
    "christian":             "Christian",
    "magic":                 "Magic",
    "adventure":             "Adventure",
    "coming-of-age":         "Coming of Age",
    "family":                "Family",
    "school":                "School Stories",
    "poetry":                "Poetry",
    "short-stories":         "Short Stories",
    "art":                   "Art",
    "travel":                "Travel",
    "business":              "Business",
    "politics":              "Politics",
    "economics":             "Economics",
    "science":               "Science",
    # regional / literary
    "latin-american":        "Latin American",
    "spanish-literature":    "Spanish Literature",
    "russian-literature":    "Russian Literature",
    "american-literature":   "American Literature",
    "british-literature":    "British Literature",
    "european-literature":   "European Literature",
    "world-literature":      "World Literature",
    "20th-century":          "20th Century",
    # age labels — kept but demoted so they don't dominate
    "young-adult":           "Young Adult",
    "middle-grade":          "Middle Grade",
    "children":              "Children's Books",
    # generics (allowed only as emergency fallback)
    "fiction":               "Fiction",
    "contemporary":          "Contemporary",
    "history":               "History",
    "historical":            "Historical",
    "nonfiction":            "Nonfiction",
    "adult":                 "Adult",
}

# Display tokens that should surface only when no preferred label is available.
# These include both mapped generics (from DIRECT_DISPLAY) and the bare-generic
# strings that result from default title-casing of un-mapped shelves such as
# `novels`, `literature`, `american`, etc.
GENERIC_DISPLAYS: frozenset[str] = frozenset({
    "Fiction", "Contemporary", "History", "Historical", "Nonfiction", "Adult",
    "Novels", "Novel", "Literature", "General", "Modern",
    "American", "English", "Translated",
})

# Show at most one age label per cluster and prefer the most specific one.
AGE_DISPLAYS_PRIORITY: tuple[str, ...] = ("Middle Grade", "Young Adult", "Children's Books")

# Heavy penalty on generics during ranking. Unlike the previous 0.35, we push
# them further down so they appear only when specifics are truly absent.
GENERIC_DISPLAY_PENALTY: float = 0.15


def _shelf_display(shelf: str) -> str:
    if shelf in FAMILY_MAP:
        return FAMILY_MAP[shelf]
    if shelf in DIRECT_DISPLAY:
        return DIRECT_DISPLAY[shelf]
    return _title_case_shelf(shelf)


def _score_displays(weighted_counts: Counter) -> tuple[dict[str, float], list[tuple[str, float, float]]]:
    """
    Aggregate shelf-level weighted counts into display-token scores.

    Returns
        raw:      dict display -> summed raw weight
        ranked:   list of (display, penalized_score, raw_score) sorted best first
    """
    raw: dict[str, float] = {}
    for shelf, weight in weighted_counts.items():
        display = _shelf_display(shelf)
        if not display:
            continue
        raw[display] = raw.get(display, 0.0) + float(weight)

    ranked_items = []
    for display, r in raw.items():
        penalized = r * (GENERIC_DISPLAY_PENALTY if display in GENERIC_DISPLAYS else 1.0)
        ranked_items.append((display, penalized, r))
    ranked_items.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return raw, ranked_items


def _apply_age_rule(chosen: list[str]) -> list[str]:
    """
    Keep at most one age label in the final label.

    `chosen` is already ordered by score, so the first age token encountered
    is the strongest signal for this cluster. That lets a children's-picture-book
    cluster keep "Children's Books" instead of being pulled to "Young Adult"
    by a fixed priority.
    """
    result: list[str] = []
    kept_age = False
    for tok in chosen:
        if tok in AGE_DISPLAYS_PRIORITY:
            if kept_age:
                continue
            kept_age = True
        result.append(tok)
    return result


def _pick_display_labels(
    ranked: list[tuple[str, float, float]],
    max_labels: int = 3,
) -> list[str]:
    """
    Pick up to `max_labels` display tokens for a cluster.

    Strategy:
        * take the top non-generic tokens (up to max_labels);
        * if fewer than 2 non-generics exist, append at most one generic
          as a graceful fallback;
        * apply the age-label rule so at most one of
          {Middle Grade, Young Adult, Children's Books} survives.
    """
    specific = [d for d, _, _ in ranked if d not in GENERIC_DISPLAYS]
    generic = [d for d, _, _ in ranked if d in GENERIC_DISPLAYS]

    chosen = specific[:max_labels]
    if len(chosen) < 2 and generic:
        chosen.append(generic[0])

    return _apply_age_rule(chosen)


def build_cluster_label(
    weighted_counts: Counter,
    cluster_id: int,
    max_labels: int = 3,
) -> tuple[str, list[str], list[tuple[str, float, float]]]:
    """
    Render a display-ready label for one cluster.

    Returns (label_str, chosen_display_tokens, ranked_display_tokens).
    `ranked_display_tokens` is exposed so the caller can run a global
    uniqueness pass that swaps later-ranked tokens when two clusters
    collide on the same label.
    """
    if not weighted_counts:
        return f"Neighborhood {cluster_id}", [], []

    _raw, ranked = _score_displays(weighted_counts)
    if not ranked:
        return f"Neighborhood {cluster_id}", [], []

    chosen = _pick_display_labels(ranked, max_labels=max_labels)
    if not chosen:
        return f"Neighborhood {cluster_id}", [], []

    return " · ".join(chosen), chosen, ranked


def resolve_duplicate_labels(
    initial: dict[int, dict],
    max_labels: int = 3,
) -> dict[int, dict]:
    """
    Global uniqueness pass.

    For each cluster whose proposed label already belongs to an earlier
    cluster (either the exact string OR the same token set in a different
    order), try swapping the lowest-ranked chosen token with the next-best
    unused alternative. As a last resort append a neutral roman-numeral suffix.
    """
    taken_labels: dict[str, int] = {}
    taken_token_sets: dict[frozenset, int] = {}
    resolved: dict[int, dict] = {}

    for cluster_id in sorted(initial.keys()):
        info = initial[cluster_id]
        chosen: list[str] = list(info["chosen"])
        ranked: list[tuple[str, float, float]] = list(info["ranked"])
        alternatives = [d for d, _, _ in ranked if d not in chosen][:6]

        def label_of(seq: list[str]) -> str:
            return " · ".join(seq)

        def collides(seq: list[str]) -> bool:
            return (
                label_of(seq) in taken_labels
                or frozenset(seq) in taken_token_sets
            )

        if not collides(chosen):
            new_chosen = chosen
        else:
            new_chosen = None
            # Try swapping each slot (last first) with the next-best alt.
            for alt in alternatives:
                for swap_idx in range(len(chosen) - 1, -1, -1):
                    trial = list(chosen)
                    trial[swap_idx] = alt
                    trial = _apply_age_rule(trial)
                    if trial and not collides(trial):
                        new_chosen = trial
                        break
                if new_chosen is not None:
                    break
            if new_chosen is None:
                # Roman suffix fallback.
                new_chosen = chosen
                base = label_of(new_chosen)
                suffix_n = 2
                while f"{base} {'I' * suffix_n}" in taken_labels:
                    suffix_n += 1
                candidate_label = f"{base} {'I' * suffix_n}"
                taken_labels[candidate_label] = cluster_id
                taken_token_sets[frozenset(new_chosen + [f"__suffix_{suffix_n}"])] = cluster_id
                resolved[cluster_id] = {"label": candidate_label, "chosen": new_chosen, "ranked": ranked}
                continue

        final_label = label_of(new_chosen)
        taken_labels[final_label] = cluster_id
        taken_token_sets[frozenset(new_chosen)] = cluster_id
        resolved[cluster_id] = {"label": final_label, "chosen": new_chosen, "ranked": ranked}

    return resolved


# ─── loader + validation ──────────────────────────────────────────────────
def load_inputs() -> tuple[np.ndarray, pd.DataFrame]:
    if not IN_EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"Missing {IN_EMBEDDINGS}. Run `python scripts/build_features.py` first."
        )
    if not IN_BOOKS.exists():
        raise FileNotFoundError(
            f"Missing {IN_BOOKS}. Run `python scripts/build_features.py` first."
        )

    embeddings = np.load(IN_EMBEDDINGS)
    books = pd.read_parquet(IN_BOOKS)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain NaN/inf values.")
    if len(books) != embeddings.shape[0]:
        raise ValueError(
            f"Row-count mismatch: search_books has {len(books)} rows but "
            f"embeddings has {embeddings.shape[0]}."
        )
    if len(books) != EXPECTED_BOOK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BOOK_COUNT} books; got {len(books)}. "
            "Regenerate search artifacts so they match the 5,000 sampled set."
        )
    for col in ("book_id", "title"):
        if col not in books.columns:
            raise ValueError(f"search_books is missing required column: {col}")

    books = books.copy()
    books["book_id"] = books["book_id"].astype(str)
    if books["book_id"].duplicated().any():
        raise ValueError("search_books contains duplicate book_id values.")
    if books["book_id"].isna().any() or (books["book_id"].str.strip() == "").any():
        raise ValueError("search_books contains empty book_id values.")

    return embeddings, books


# ─── orchestration ───────────────────────────────────────────────────────
def build() -> None:
    embeddings, books = load_inputs()
    print(f"Loaded embeddings: shape={embeddings.shape}  books: {len(books)}")

    result = spherical_kmeans(
        embeddings,
        k=N_CLUSTERS,
        max_iter=MAX_ITER,
        tol=TOL,
        seed=RANDOM_SEED,
    )
    print(
        f"Spherical k-means: iterations={result.iterations_run}  "
        f"converged={result.converged}  final_inertia={result.inertia:.4f}"
    )

    labels = result.labels
    cluster_sizes = np.bincount(labels, minlength=N_CLUSTERS)
    if (cluster_sizes == 0).any():
        empty_ids = np.flatnonzero(cluster_sizes == 0).tolist()
        raise RuntimeError(
            f"Final clustering produced empty clusters: {empty_ids}. "
            "Re-run with a different seed or lower N_CLUSTERS."
        )
    if int(cluster_sizes.sum()) != EXPECTED_BOOK_COUNT:
        raise RuntimeError(
            f"Cluster sizes sum to {int(cluster_sizes.sum())}, expected "
            f"{EXPECTED_BOOK_COUNT}."
        )

    similarity, cluster_rank, is_representative = rank_cluster_representatives(
        embeddings,
        labels=labels,
        centroids=result.centroids,
        top_k=REPRESENTATIVES_PER_CLUSTER,
    )
    if not np.all(np.isfinite(similarity)):
        raise RuntimeError("Centroid similarities contain NaN/inf after ranking.")

    # Cluster labeling from cleaned shelves.
    # Two phases: (1) propose labels per cluster using weighted shelf counts,
    # resolved into display-token families; (2) global uniqueness pass across
    # all clusters so no two clusters share the same label (or same token set
    # in a different order).
    books["_cleaned_shelves"] = books["popular_shelves"].fillna("").apply(clean_shelves)
    cleaned_shelves_list = books["_cleaned_shelves"].tolist()

    initial: dict[int, dict] = {}
    cluster_top_cleaned: dict[int, list[str]] = {}
    cluster_weighted_counts: dict[int, Counter] = {}

    for cluster_id in range(N_CLUSTERS):
        member_indices = np.flatnonzero(labels == cluster_id)
        weighted_counts = weighted_shelf_counts(
            cleaned_shelves_list, member_indices, cluster_rank
        )
        _label, chosen, ranked = build_cluster_label(weighted_counts, cluster_id)
        initial[cluster_id] = {"chosen": chosen, "ranked": ranked}
        cluster_weighted_counts[cluster_id] = weighted_counts
        # Raw top cleaned shelves (before display resolution) for the audit JSON.
        cluster_top_cleaned[cluster_id] = [
            shelf for shelf, _ in weighted_counts.most_common(8)
        ]

    # Resolve collisions globally.
    resolved = resolve_duplicate_labels(initial)
    cluster_labels: dict[int, str] = {cid: info["label"] for cid, info in resolved.items()}
    cluster_chosen: dict[int, list[str]] = {cid: info["chosen"] for cid, info in resolved.items()}
    cluster_ranked: dict[int, list[tuple[str, float, float]]] = {
        cid: info["ranked"] for cid, info in resolved.items()
    }

    # Assemble the per-book output row set.
    out = pd.DataFrame({
        "book_id":              books["book_id"].values,
        "title":                books["title"].astype(str).values,
        "cluster_id":           labels.astype(np.int32),
        "cluster_label":        [cluster_labels[int(c)] for c in labels],
        "cluster_size":         cluster_sizes[labels].astype(np.int32),
        "cluster_rank":         cluster_rank.astype(np.int32),
        "centroid_similarity":  similarity.astype(np.float32),
        "centroid_distance":    (1.0 - similarity).astype(np.float32),
        "is_representative":    is_representative.astype(bool),
    })

    if len(out) != EXPECTED_BOOK_COUNT:
        raise RuntimeError(
            f"Output row count {len(out)} != expected {EXPECTED_BOOK_COUNT}."
        )
    if out["cluster_id"].isna().any():
        raise RuntimeError("Some cluster_id values are null.")
    if out["book_id"].isna().any() or (out["book_id"].astype(str).str.strip() == "").any():
        raise RuntimeError("Some book_id values are empty in the output.")
    if out["book_id"].duplicated().any():
        raise RuntimeError("Duplicate book_id values in output.")

    # Cluster-level summary section.
    cluster_summaries = []
    for cluster_id in range(N_CLUSTERS):
        rep_titles = (
            out.loc[
                (out["cluster_id"] == cluster_id) & (out["cluster_rank"] <= REPRESENTATIVES_PER_CLUSTER),
                "title",
            ]
            .tolist()
        )
        ranked = cluster_ranked[cluster_id]
        cluster_summaries.append({
            "cluster_id": int(cluster_id),
            "cluster_label": cluster_labels[cluster_id],
            "cluster_size": int(cluster_sizes[cluster_id]),
            "selected_label_families": cluster_chosen[cluster_id],
            "top_family_scores": [
                {"label": display, "score": round(float(raw_score), 2)}
                for display, _penalized, raw_score in ranked[:8]
            ],
            "top_cleaned_shelves": cluster_top_cleaned[cluster_id],
            "representative_titles": rep_titles,
        })

    if len(cluster_summaries) != N_CLUSTERS:
        raise RuntimeError(
            f"Summary has {len(cluster_summaries)} clusters; expected {N_CLUSTERS}."
        )

    summary = {
        "n_books": int(len(out)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_clusters": N_CLUSTERS,
        "algorithm": "spherical_kmeans",
        "initialization": "kmeans++",
        "distance_metric": "cosine_distance",
        "max_iter": MAX_ITER,
        "iterations_run": int(result.iterations_run),
        "converged": bool(result.converged),
        "final_inertia": float(result.inertia),
        "min_cluster_size": int(cluster_sizes.min()),
        "median_cluster_size": int(np.median(cluster_sizes)),
        "max_cluster_size": int(cluster_sizes.max()),
        "random_seed": RANDOM_SEED,
        "outputs": [
            "data/artifacts/book_clusters.parquet",
            "data/artifacts/book_cluster_summary.json",
        ],
        "clusters": cluster_summaries,
    }

    OUT_CLUSTERS.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_CLUSTERS, index=False)
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUT_CLUSTERS}  rows={len(out)}")
    print(f"Wrote {OUT_SUMMARY}")
    print(
        f"cluster size: min={summary['min_cluster_size']}  "
        f"median={summary['median_cluster_size']}  "
        f"max={summary['max_cluster_size']}"
    )


if __name__ == "__main__":
    build()
