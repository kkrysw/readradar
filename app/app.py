"""
ReadRadar — Streamlit app.

Two pages:
    1. Thematic Search — semantic search over the 5,000 sampled books.
       Each result card can be opened for details (controversy tags +
       judgment + metadata) and added to favorites.
    2. Recommendations — favorites list + recency-weighted recommendations.

Artifacts consumed (produced by the backend pipeline):
    data/artifacts/ui_books_cache.parquet     (metadata + top_tags + judgment)
    data/artifacts/rec_embeddings.npy
    data/artifacts/rec_embeddings_ids.json
    data/artifacts/search_books.parquet       (via src.search)
    data/artifacts/search_embeddings.npy      (via src.search)
"""

from __future__ import annotations

import ast
import html
import json
import re
import sys
from pathlib import Path

# Ensure repo root is importable so `src.*` resolves under `streamlit run app/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

import src.search as search
from src.recommend import recommend

# Streamlit puts app/ (the script's directory) on sys.path, so styles.py is importable by name.
from styles import CSS


# ─── page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="ReadRadar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)


# ─── paths ─────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
ARTIFACTS_DIR = DATA_DIR / "artifacts"
PROCESSED_DIR = DATA_DIR / "processed"

UI_CACHE_PATH       = ARTIFACTS_DIR / "ui_books_cache.parquet"
REC_EMBEDDINGS_PATH = ARTIFACTS_DIR / "rec_embeddings.npy"
REC_IDS_PATH        = ARTIFACTS_DIR / "rec_embeddings_ids.json"
BOOKS_PATH          = PROCESSED_DIR / "books.parquet"

REQUIRED_PATHS = [UI_CACHE_PATH, REC_EMBEDDINGS_PATH, REC_IDS_PATH, BOOKS_PATH]

TABS = {
    "search": "🔍 Thematic Search",
    "recs":   "📚 Favorites & Recommendations",
}

DETAIL_COLUMNS = [
    "book_id",
    "title",
    "description",
    "average_rating",
    "ratings_count",
    "text_reviews_count",
    "language_code",
    "image_url",
    "publication_year",
    "num_pages",
    "publisher",
    "top_tags",
    "overall_judgment",
]


# ─── data loading ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    """Load the UI master table, recommendation artifacts, and books lookup."""
    ui_books = pd.read_parquet(UI_CACHE_PATH)
    ui_books["book_id"] = ui_books["book_id"].astype(str)

    for col in ["average_rating", "ratings_count", "text_reviews_count", "publication_year", "num_pages"]:
        if col in ui_books.columns:
            ui_books[col] = pd.to_numeric(ui_books[col], errors="coerce")
    for col in ["ratings_count", "text_reviews_count"]:
        if col in ui_books.columns:
            ui_books[col] = ui_books[col].fillna(0).astype(int)

    rec_embeddings = np.load(REC_EMBEDDINGS_PATH)
    with open(REC_IDS_PATH, "r") as f:
        rec_book_ids = [str(bid) for bid in json.load(f)]

    rec_books_df = pd.read_parquet(
        BOOKS_PATH,
        columns=["book_id", "title", "average_rating", "ratings_count"],
    )
    rec_books_df["book_id"] = rec_books_df["book_id"].astype(str)
    sampled_ids = set(rec_book_ids)
    rec_books_df = rec_books_df[rec_books_df["book_id"].isin(sampled_ids)].copy()
    rec_books_df["average_rating"] = pd.to_numeric(rec_books_df["average_rating"], errors="coerce")
    rec_books_df["ratings_count"] = (
        pd.to_numeric(rec_books_df["ratings_count"], errors="coerce").fillna(0).astype(int)
    )

    return ui_books, rec_embeddings, rec_book_ids, rec_books_df


def safe_load():
    if not all(p.exists() for p in REQUIRED_PATHS):
        return None, None, None, None
    return load_data()


# ─── small helpers ─────────────────────────────────────────────────────────
def stars(rating) -> str:
    rating = float(rating or 0)
    full = max(0, min(int(round(rating)), 5))
    return "★" * full + "☆" * (5 - full)


def parse_tags(value) -> list[str]:
    """Read top_tags from a DataFrame cell that could be list / JSON string / repr-string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
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


def has_real_value(value) -> bool:
    """True iff the value is worth showing (not None/NaN/0/empty)."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    return str(value).strip() not in {"", "0"}


def title_initials(title: str) -> str:
    """Return one or two upper-case initials for the cover placeholder."""
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", str(title or "")).strip()
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return "•"
    if len(tokens) == 1:
        return tokens[0][:2].upper()
    return (tokens[0][0] + tokens[1][0]).upper()


def _is_real_cover_url(url: str) -> bool:
    """
    Goodreads served 'nophoto' URLs for books with no cover art. Detect those
    so we substitute the designed placeholder. Real covers (on
    `images.gr-assets.com` or elsewhere) are always used; if such a URL 404s,
    the <img onerror> handler hides the broken image and the designed
    placeholder layered behind it becomes visible.
    """
    if not url:
        return False
    u = url.lower().strip()
    if "nophoto" in u:
        return False
    # Match the static/placeholder host with delimiters so we don't also match
    # `images.gr-assets.com`, which legitimately hosts real cover art.
    if "//s.gr-assets.com/" in u:
        return False
    return True


def cover_html(image_url: str, title: str) -> str:
    """
    Book-cover HTML with an inline designed placeholder. Real images are
    layered over the placeholder; an onerror handler hides the image if
    the URL fails to load, so the placeholder remains visible.
    """
    initials = html.escape(title_initials(title))
    safe_title = html.escape(str(title or ""))
    placeholder = (
        '<div class="cover-placeholder">'
        f'<div class="ph-initials">{initials}</div>'
        '<div class="ph-label">ReadRadar</div>'
        "</div>"
    )
    if _is_real_cover_url(image_url):
        safe_url = html.escape(str(image_url), quote=True)
        img_tag = (
            f'<img class="cover-img" src="{safe_url}" alt="{safe_title}" '
            f'loading="lazy" referrerpolicy="no-referrer" '
            f'onerror="this.style.display=\'none\'" />'
        )
    else:
        img_tag = ""
    return f'<div class="cover-wrap">{placeholder}{img_tag}</div>'


def enrich_with_details(results_df: pd.DataFrame, ui_books: pd.DataFrame) -> pd.DataFrame:
    """Left-merge a search/rec result set with the UI master table on book_id."""
    if results_df is None or results_df.empty:
        return results_df
    results_df = results_df.copy()
    results_df["book_id"] = results_df["book_id"].astype(str)
    detail_lookup = ui_books[[c for c in DETAIL_COLUMNS if c in ui_books.columns]].drop_duplicates("book_id")
    return results_df.merge(detail_lookup, on="book_id", how="left", sort=False, suffixes=("", "_ui"))


# ─── modal state control ───────────────────────────────────────────────────
def clear_modal() -> None:
    """Reset the detail-modal state. Called on nav/tab/page/search transitions."""
    st.session_state.modal_book_id = None


def _on_tab_change() -> None:
    """Radio on_change: sync active_tab and close any open modal."""
    st.session_state.active_tab = st.session_state.nav_tab
    clear_modal()


# ─── recommendation ────────────────────────────────────────────────────────
def refresh_recs(
    rec_embeddings: np.ndarray,
    rec_book_ids: list[str],
    rec_books_df: pd.DataFrame,
    ui_books: pd.DataFrame,
) -> None:
    """Recompute recommendations from the current favorites list."""
    liked_ids = list(st.session_state.liked_books.keys())

    if not liked_ids:
        recs = (
            rec_books_df[["book_id", "title", "average_rating", "ratings_count"]]
            .sort_values(["average_rating", "ratings_count"], ascending=[False, False])
            .head(5)
            .reset_index(drop=True)
        )
        recs.insert(0, "rank", recs.index + 1)
        recs["similarity"] = None
        st.session_state.recs_mode = "top_rated"
    else:
        recs = recommend(liked_ids, rec_embeddings, rec_book_ids, rec_books_df, top_n=5)
        st.session_state.recs_mode = "personalized"

    recs["book_id"] = recs["book_id"].astype(str)
    recs = enrich_with_details(recs, ui_books)
    # Keep cosine similarity in [0..1] so the modal scales it once.
    recs["score"] = pd.to_numeric(recs.get("similarity"), errors="coerce")
    st.session_state.recs = recs


# ─── cards ─────────────────────────────────────────────────────────────────
def book_card(row: pd.Series, source: str) -> None:
    """Render a single result row as a card with cover, meta, tags, actions."""
    bid = str(row["book_id"])
    title = str(row.get("title", "Unknown Title") or "Unknown Title")
    avg = float(row.get("average_rating") or 0)
    ratings_count = int(row.get("ratings_count") or 0)

    metric_html = ""
    if source == "search":
        score = float(row.get("score") or 0)
        if score <= 1:
            score *= 100
        metric_html = f'<span class="bc-score">Relevance · {score:.0f}</span>'
    elif source == "rec":
        sim = row.get("similarity")
        if sim is not None and not (isinstance(sim, float) and pd.isna(sim)):
            metric_html = f'<span class="bc-score">Match · {float(sim):.0%}</span>'

    desc = str(row.get("description", "") or "").strip()
    desc_preview = desc[:240] + "…" if len(desc) > 240 else desc

    tags = parse_tags(row.get("top_tags"))
    tag_html = ""
    if tags:
        chips = "".join(f'<span class="bc-tag">{html.escape(t)}</span>' for t in tags[:4])
        tag_html = f'<div class="bc-tags">{chips}</div>'

    image_url = str(row.get("image_url", "") or "").strip()

    col_img, col_main, col_actions = st.columns(
        [1.0, 5.2, 1.3], gap="medium", vertical_alignment="center"
    )

    with col_img:
        st.markdown(cover_html(image_url, title), unsafe_allow_html=True)

    with col_main:
        meta_bits = [f'<span class="bc-stars">{stars(avg)}</span>']
        if has_real_value(avg):
            meta_bits.append(f"{avg:.2f}")
        if ratings_count:
            meta_bits.append(f"{ratings_count:,} ratings")
        meta_line = '<span class="dot">·</span>'.join(
            f'<span>{m}</span>' for m in meta_bits
        )

        # Build the card as one contiguous HTML string with no leading
        # whitespace or blank lines. Streamlit runs markdown on this, and a
        # blank line between indented HTML blocks causes the next block to
        # be parsed as a code block (escaping the tags as literal text).
        card_parts = [
            '<div class="book-card">',
            f'<div class="bc-title">{html.escape(title)}</div>',
            f'<div class="bc-meta">{meta_line}{metric_html}</div>',
        ]
        if desc_preview:
            card_parts.append(
                f'<div class="bc-desc">{html.escape(desc_preview)}</div>'
            )
        if tag_html:
            card_parts.append(tag_html)
        card_parts.append("</div>")
        st.markdown("".join(card_parts), unsafe_allow_html=True)

    with col_actions:
        if st.button("Details", key=f"open_{source}_{bid}", type="secondary"):
            st.session_state.modal_book_id = bid
            st.rerun()

        if bid not in st.session_state.liked_books:
            if st.button("+ Favorite", key=f"add_{source}_{bid}", type="primary"):
                # Clear any stale modal before reruns so a previously-opened
                # detail dialog (dismissed via the browser X) does not resurface.
                clear_modal()
                st.session_state.liked_books[bid] = title
                refresh_recs(rec_embeddings, rec_book_ids, rec_books_df, books)
                st.rerun()
        else:
            st.markdown(
                "<div style='text-align:center;font-size:0.78rem;color:var(--accent);"
                "letter-spacing:0.08em;text-transform:uppercase;margin-top:0.5rem;'>"
                "✓ In favorites</div>",
                unsafe_allow_html=True,
            )


# ─── detail modal ──────────────────────────────────────────────────────────
@st.dialog("Book Details", width="large")
def show_modal(book_id: str) -> None:
    """Render the detail view for a single book."""
    results = (
        st.session_state.recs
        if st.session_state.active_tab == "recs"
        else st.session_state.thematic_results
    )
    if results is None or results.empty:
        clear_modal()
        st.rerun()

    match = results[results["book_id"].astype(str) == str(book_id)]
    if match.empty:
        clear_modal()
        st.rerun()
    row = match.iloc[0]

    title = str(row.get("title", "") or "")
    desc = str(row.get("description", "") or "").strip()
    avg = float(row.get("average_rating") or 0)
    ratings_count = int(row.get("ratings_count") or 0)
    image_url = str(row.get("image_url", "") or "").strip()
    tags = parse_tags(row.get("top_tags"))
    overall_judgment = str(row.get("overall_judgment", "") or "").strip()

    raw_score = row.get("score") or 0
    try:
        raw_score = float(raw_score)
    except (TypeError, ValueError):
        raw_score = 0.0
    if raw_score <= 1:
        raw_score *= 100
    metric_label = "Match" if st.session_state.active_tab == "recs" else "Relevance"

    col_cover, col_body = st.columns([1, 2.4], gap="large")

    with col_cover:
        st.markdown(cover_html(image_url, title), unsafe_allow_html=True)

        bid = str(row["book_id"])
        st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
        if bid not in st.session_state.liked_books:
            if st.button("+ Add to favorites", key=f"modal_add_{bid}", type="primary", use_container_width=True):
                st.session_state.liked_books[bid] = title
                refresh_recs(rec_embeddings, rec_book_ids, rec_books_df, books)
                st.rerun()
        else:
            st.markdown(
                "<div style='text-align:center;font-size:0.78rem;color:var(--accent);"
                "letter-spacing:0.08em;text-transform:uppercase;padding:0.55rem 0;'>"
                "✓ In favorites</div>",
                unsafe_allow_html=True,
            )

    with col_body:
        st.markdown(f'<div class="dm-title">{html.escape(title)}</div>', unsafe_allow_html=True)

        sub_parts = [f'<span class="bc-stars">{stars(avg)}</span>']
        if has_real_value(avg):
            sub_parts.append(f'<span>{avg:.2f}</span>')
        if ratings_count:
            sub_parts.append(f'<span>{ratings_count:,} ratings</span>')
        if raw_score > 0:
            sub_parts.append(f'<span class="bc-score">{metric_label} · {raw_score:.0f}</span>')
        sub_line = '<span class="dot">·</span>'.join(sub_parts)
        st.markdown(f'<div class="dm-sub">{sub_line}</div>', unsafe_allow_html=True)

        if tags:
            chips = "".join(f'<span class="bc-tag">{html.escape(t)}</span>' for t in tags)
            st.markdown(
                f'<div class="dm-section-label">Top tags</div>'
                f'<div class="bc-tags">{chips}</div>',
                unsafe_allow_html=True,
            )

        if overall_judgment:
            st.markdown(
                '<div class="dm-section-label">Overall reader judgment</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="dm-judgment">{html.escape(overall_judgment)}</div>',
                unsafe_allow_html=True,
            )

        if desc:
            st.markdown('<div class="dm-section-label">Description</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="dm-desc">{html.escape(desc)}</div>', unsafe_allow_html=True)

        meta_items: list[tuple[str, str]] = []
        year = row.get("publication_year")
        pages = row.get("num_pages")
        publisher = row.get("publisher")
        language_code = row.get("language_code")
        if has_real_value(year):
            try:
                meta_items.append(("Published", str(int(float(year)))))
            except (TypeError, ValueError):
                meta_items.append(("Published", str(year)))
        if has_real_value(pages):
            try:
                meta_items.append(("Pages", f"{int(float(pages))}"))
            except (TypeError, ValueError):
                meta_items.append(("Pages", str(pages)))
        if has_real_value(publisher):
            meta_items.append(("Publisher", str(publisher)))
        if has_real_value(language_code):
            meta_items.append(("Language", str(language_code)))

        if meta_items:
            st.markdown('<div class="dm-section-label">Details</div>', unsafe_allow_html=True)
            grid = "".join(
                f'<div class="dm-meta-item"><span class="label">{html.escape(lbl)}</span>{html.escape(val)}</div>'
                for lbl, val in meta_items
            )
            st.markdown(f'<div class="dm-meta-grid">{grid}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    if st.button("Close", key=f"modal_close_{bid}", type="secondary"):
        clear_modal()
        st.rerun()


# ─── session state ─────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "active_tab":         "search",
        "nav_tab":            TABS["search"],
        "search_page":        1,
        "results_per_page":   10,
        "modal_book_id":      None,
        "thematic_query":     "",
        "thematic_results":   None,
        "thematic_searched":  False,
        "liked_books":        {},  # insertion order = recency (newest last)
        "recs":               pd.DataFrame(),
        "recs_mode":          "top_rated",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── UI ────────────────────────────────────────────────────────────────────
init_session_state()

st.markdown(
    """
<div class="hero">
    <h1><span class="radar-dot"></span>ReadRadar</h1>
    <div class="subtitle">Thematic Literature Explorer · UCSD Book Graph</div>
</div>
""",
    unsafe_allow_html=True,
)

books, rec_embeddings, rec_book_ids, rec_books_df = safe_load()

if books is None:
    st.error(
        "**Artifacts not found.** Generate the search, recommendation, and UI "
        "cache artifacts first. See the README for the official pipeline commands."
    )
    st.stop()

if st.session_state.recs.empty:
    refresh_recs(rec_embeddings, rec_book_ids, rec_books_df, books)

# ─── top navigation ────────────────────────────────────────────────────────
st.radio(
    "Navigation",
    list(TABS.values()),
    key="nav_tab",
    horizontal=True,
    label_visibility="collapsed",
    on_change=_on_tab_change,
)
tab_value_to_key = {v: k for k, v in TABS.items()}
st.session_state.active_tab = tab_value_to_key.get(st.session_state.nav_tab, "search")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — THEMATIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "search":
    st.markdown(
        '<div class="section-label">Describe what you\'re looking for</div>',
        unsafe_allow_html=True,
    )

    with st.form("thematic_search_form", clear_on_submit=False):
        col_q, col_btn = st.columns([5, 1], vertical_alignment="bottom")
        with col_q:
            query = st.text_input(
                "query",
                placeholder="e.g. dark academia with found family in a Victorian setting…",
                label_visibility="collapsed",
                key="thematic_query",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.form_submit_button("Search")

    if search_btn:
        clear_modal()
        st.session_state.thematic_searched = True
        st.session_state.search_page = 1
        if query.strip():
            with st.spinner("Searching…"):
                raw_results = search.thematic_search(query.strip(), top_k=100)
            st.session_state.thematic_results = enrich_with_details(raw_results, books)
        else:
            st.session_state.thematic_results = None

    if st.session_state.thematic_searched:
        if not st.session_state.thematic_query.strip():
            st.warning("Please enter a query.")
        elif st.session_state.thematic_results is None or st.session_state.thematic_results.empty:
            st.info("No matches found.")
        else:
            results = st.session_state.thematic_results
            total = len(results)
            per_page = st.session_state.results_per_page
            total_pages = max(1, (total + per_page - 1) // per_page)
            page = min(st.session_state.search_page, total_pages)
            start = (page - 1) * per_page
            end = start + per_page

            st.markdown(
                f'<div class="result-count">{total} results · page {page} of {total_pages}</div>',
                unsafe_allow_html=True,
            )

            for _, row in results.iloc[start:end].iterrows():
                book_card(row, source="search")

            col_prev, col_mid, col_next = st.columns([1, 2, 1])
            with col_prev:
                if page > 1:
                    if st.button("← Previous", key="search_prev_page", type="secondary"):
                        clear_modal()
                        st.session_state.search_page = page - 1
                        st.rerun()
            with col_mid:
                st.markdown(
                    f'<div class="page-indicator">Page {page} / {total_pages}</div>',
                    unsafe_allow_html=True,
                )
            with col_next:
                if page < total_pages:
                    if st.button("Next →", key="search_next_page", type="secondary"):
                        clear_modal()
                        st.session_state.search_page = page + 1
                        st.rerun()

    with st.expander("Search tips"):
        st.markdown(
            """
<div class="tips-grid">
    <div class="tips-card">
        <div class="tc-label">Themes</div>
        <span class="tc-example">redemption arc</span>
        <span class="tc-example">morally grey protagonist</span>
        <span class="tc-example">slow burn romance</span>
    </div>
    <div class="tips-card">
        <div class="tc-label">Settings</div>
        <span class="tc-example">post-apocalyptic</span>
        <span class="tc-example">Victorian London</span>
        <span class="tc-example">high fantasy court</span>
    </div>
    <div class="tips-card">
        <div class="tc-label">Moods</div>
        <span class="tc-example">cozy mystery</span>
        <span class="tc-example">dark and atmospheric</span>
        <span class="tc-example">hopeful</span>
    </div>
    <div class="tips-card">
        <div class="tc-label">Subjects</div>
        <span class="tc-example">time travel</span>
        <span class="tc-example">dragons</span>
        <span class="tc-example">magic academy</span>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — FAVORITES + RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "recs":
    st.markdown('<div class="section-title">Your reading list</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Ordered oldest → newest. Later favorites weigh more in recommendations.</div>',
        unsafe_allow_html=True,
    )

    liked_items = list(st.session_state.liked_books.items())
    if liked_items:
        for idx, (bid, title) in enumerate(liked_items, start=1):
            row_data = rec_books_df[rec_books_df["book_id"] == bid]
            avg = float(row_data["average_rating"].values[0]) if not row_data.empty else 0.0
            col_info, col_btn = st.columns([12, 1], vertical_alignment="center")
            with col_info:
                st.markdown(
                    f"""
<div class="fav-row">
    <div style="display:flex;align-items:center;">
        <span class="fav-order">{idx}</span>
        <div>
            <div class="fav-title">{html.escape(title)}</div>
            <div class="fav-meta"><span class="bc-stars">{stars(avg)}</span> &nbsp;{avg:.2f}</div>
        </div>
    </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_btn:
                if st.button("✕", key=f"rm_{bid}", help="Remove from favorites"):
                    clear_modal()
                    del st.session_state.liked_books[bid]
                    refresh_recs(rec_embeddings, rec_book_ids, rec_books_df, books)
                    st.rerun()
    else:
        st.markdown(
            """
<div class="empty-state">
    <div class="es-icon">📖</div>
    <div class="es-title">Your reading list is empty</div>
    <div class="es-body">Add books from the search page to build your personal recommendations.</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Recommended for you</div>', unsafe_allow_html=True)
    if st.session_state.recs_mode == "top_rated":
        st.markdown(
            '<div class="section-label">Showing top-rated picks · add favorites for personalized recommendations</div>',
            unsafe_allow_html=True,
        )

    for _, row in st.session_state.recs.iterrows():
        book_card(row, source="rec")


# ─── open the detail modal only when explicitly triggered ──────────────────
if st.session_state.modal_book_id:
    show_modal(st.session_state.modal_book_id)


# ─── footer ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="data-credit">
    Data: <a href="https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home" target="_blank">
    UCSD Book Graph</a> (Goodreads snapshot, 2017) · Wan et al., 2018
</div>
""",
    unsafe_allow_html=True,
)
