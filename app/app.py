"""

ReadRadar — Thematic Literature Explorer
Streamlit prototype using UCSD Book Graph (fantasy/paranormal subset)
Data source: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
"""
import json
import sys
from pathlib import Path

# ensure repo root is on sys.path so src.* modules resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

import src.search as search
from src.recommendation_step2_algorithm import recommend
import src.controversy_finalize as controversy

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ReadRadar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── inject custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── base: warm off-white light theme ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #f7f3ee !important;
    color: #2c2420 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #f7f3ee;
}

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1080px; }

/* ── hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.8rem;
    border-bottom: 2px solid #e0d5c5;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #1a1210;
    margin: 0;
}
    .hero .subtitle {
    font-size: 0.8rem;
    color: #9a8878;
    margin-top: 0.5r    em;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.hero .radar-dot {
    display: inline-blo    ck;
    width: 10px; height: 10px;
    background: #b87333;
    border-radius: 50%;
    margin-right: 10px;
    vertical-align: middle;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.3; transform: scale(1.5); }
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ede6da !important;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #d5c9b8;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #7a6655 !important;
    background: transparent !important;
    border-radius: 7px !important;
    padding: 0.5rem 1.3rem !important;
    border: none !important;
    transition: background 0.15s, color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #e0d5c5 !important;
    color: #3a2a1a !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #b87333 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
    border: none !important;
}
/* tab panel itself */
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.2rem !important;
}

/* ── inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 1.5px solid #d5c9b8 !important;
    border-radius: 8px !important;
    color: #2c2420 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #b87333 !important;
    box-shadow: 0 0 0 2px rgba(184,115,51,0.15) !important;
}
/* selectbox */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1.5px solid #d5c9b8 !important;
    border-radius: 8px !important;
    color: #2c2420 !important;
}

/* ── buttons ── */
.stButton > button {
    background: #b87333 !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0.55rem 1.5rem !important;
    transition: background 0.2s, transform 0.1s !important;
}

.stButton > button:hover {
    background: #9e6228 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── general text colours ── */
p, li, span, div {
    color: #2c2420;
}
label, .stMarkdown p {
    color: #2c2420 !important;
}


/* ── book card ── */
.book-card {
    background: #ffffff;
    border: 1px solid #e0d5c5;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
}
.book-card:hover {
    border-color: #b87333;
    box-shadow: 0 3px 10px rgba(184,115,51,0.12);
    transform: translateY(-2px);
}
.book-card .title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1210;
    margin-bottom: 0.2rem;
}
.book-card .meta {
    font-size: 0.78rem;
    color: #9a8878;
    margin-bottom: 0.5rem;
}
.book-card .desc {
    font-size: 0.875rem;
    color: #4a3c30;
    line-height: 1.6;
}
.book-card .score-pill {
    display: inline-block;
    background: #fdf6ec;
    border: 1px solid #e0c9a0;
    color: #b87333;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-top: 0.6rem;
}
.book-card .stars {
    color: #c8920a;
    font-size: 0.88rem;
}

/* ── controversy meter ── */
.controversy-meter {
    background: #ffffff;
    border: 1px solid #e0d5c5;
    border-radius: 12px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.controversy-meter .score-number {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    color: #b87333;
    line-height: 1;
}
.controversy-meter .score-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9a8878;
    margin-top: 0.3rem;
}
.controversy-meter .bar-bg {
    background: #ede6da;
    border-radius: 4px;
    height: 7px;
    margin-top: 1rem;
    overflow: hidden;
}
.controversy-meter .bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #e8c070, #b87333);
}

/* ── pros/cons ── */
.pro-item {
    background: #f2faf5;
    border: 1px solid #c3e0ce;
    border-left: 4px solid #3a8c5c;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.875rem;
    line-height: 1.55;
    color: #2a4a36;
}
.con-item {
    background: #fdf2f2;
    border: 1px solid #e0c3c3;
    border-left: 4px solid #a03030;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.875rem;
    line-height: 1.55;
    color: #4a2a2a;
}

/* ── section label ── */
.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #b0a090;
    font-weight: 500;
    margin-bottom: 0.6rem;
    margin-top: 1.2rem;
}

/* ── expander ── */
.stExpander {
    background: #ffffff !important;
    border: 1px solid #e0d5c5 !important;
    border-radius: 8px !important;
}
.stExpander summary {
    color: #4a3c30 !important;
    font-size: 0.88rem !important;
}

/* ── info / warning banners ── */
.stAlert {
    border-radius: 8px !important;
}

/* ── footnote ── */
.data-credit {
    text-align: center;
    font-size: 0.72rem;
    color: #b0a090;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #e0d5c5;
    letter-spacing: 0.05em;
}
.data-credit a { color: #9a7a50; text-decoration: none; }
.data-credit a:hover { color: #b87333; text-decoration: underline; }

/* ── Tittle Trigger ── */
/* Plain clickable title button
   Apply to all tertiary buttons globally.
   Only the book title should use type="tertiary". */

.stButton > button[kind="tertiary"] {
    background: transparent !important;
    color: #1a1210 !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 0 0.35rem 0 !important;
    text-align: left !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    line-height: 1.3 !important;
    min-height: unset !important;
}

.stButton > button[kind="tertiary"]:hover,
.stButton > button[kind="tertiary"]:focus,
.stButton > button[kind="tertiary"]:active {
    background: transparent !important;
    color: #b87333 !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── data loading ───────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ARTIFACTS_DIR = DATA_DIR / "artifacts"
PROCESSED_DIR = DATA_DIR / "processed"

def _sanitize_object_col(series):
    """Convert nested numpy arrays in object columns to plain Python lists.
    Streamlit's cache hasher cannot handle numpy.ndarray as cell values
    (raises: TypeError: unhashable type: 'numpy.ndarray')."""
    def _to_pyobj(val):
        if val is None:
            return val
        if isinstance(val, np.ndarray):
            result = []
            for item in val.tolist():
                if isinstance(item, dict):
                    result.append({k: (v.item() if isinstance(v, np.generic) else v)
                                   for k, v in item.items()})
                else:
                    result.append(item)
            return result
        return val
    return series.apply(_to_pyobj)



@st.cache_data(show_spinner=False)
def load_data():
    # --------------------------------------------------
    # 1. Main UI cache for search + modal detail
    # --------------------------------------------------
    ui_books = pd.read_parquet(ARTIFACTS_DIR / "ui_books_cache.parquet")
    ui_books["book_id"] = ui_books["book_id"].astype(str)

    for col in ["average_rating", "ratings_count", "text_reviews_count", "publication_year", "num_pages"]:
        if col in ui_books.columns:
            ui_books[col] = pd.to_numeric(ui_books[col], errors="coerce")

    if "ratings_count" in ui_books.columns:
        ui_books["ratings_count"] = ui_books["ratings_count"].fillna(0).astype(int)

    if "text_reviews_count" in ui_books.columns:
        ui_books["text_reviews_count"] = ui_books["text_reviews_count"].fillna(0).astype(int)

    # --------------------------------------------------
    # 2. Recommendation embeddings + matching sampled ids
    # --------------------------------------------------
    rec_embeddings = np.load(ARTIFACTS_DIR / "rec_embeddings.npy")

    with open(ARTIFACTS_DIR / "rec_embeddings_ids.json", "r") as f:
        rec_book_ids = json.load(f)

    rec_book_ids = [str(bid) for bid in rec_book_ids]
    sampled_ids = set(rec_book_ids)

    # --------------------------------------------------
    # 3. Recommendation metadata from full books.parquet
    #    Keep only the sampled 5000 books
    # --------------------------------------------------
    rec_books_df = pd.read_parquet(
        PROCESSED_DIR / "books.parquet",
        columns=["book_id", "title", "average_rating", "ratings_count"]
    )
    rec_books_df["book_id"] = rec_books_df["book_id"].astype(str)
    rec_books_df = rec_books_df[rec_books_df["book_id"].isin(sampled_ids)].copy()

    rec_books_df["average_rating"] = pd.to_numeric(rec_books_df["average_rating"], errors="coerce")
    rec_books_df["ratings_count"] = (
        pd.to_numeric(rec_books_df["ratings_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------
    # 4. Full metadata lookup for recommendation descriptions
    # --------------------------------------------------
    full_books = pd.read_parquet(
        PROCESSED_DIR / "books.parquet",
        columns=["book_id", "title", "description", "average_rating", "ratings_count"]
    )
    full_books["book_id"] = full_books["book_id"].astype(str)

    full_books["average_rating"] = pd.to_numeric(full_books["average_rating"], errors="coerce")
    full_books["ratings_count"] = (
        pd.to_numeric(full_books["ratings_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------
    # 5. 3D universe data for future recommendation page
    # --------------------------------------------------
    universe_df = None
    universe_path = ARTIFACTS_DIR / "read_universe_3d.parquet"
    if universe_path.exists():
        universe_df = pd.read_parquet(universe_path)
        universe_df["book_id"] = universe_df["book_id"].astype(str)


    return (
        ui_books,
        rec_embeddings,
        rec_book_ids,
        rec_books_df,
        full_books,
        universe_df,
    )

def safe_load():
    required = [
        ARTIFACTS_DIR / "search_books.parquet",
        ARTIFACTS_DIR / "ui_books_cache.parquet",
        PROCESSED_DIR / "books.parquet",
        ARTIFACTS_DIR / "rec_embeddings.npy",
        ARTIFACTS_DIR / "rec_embeddings_ids.json",
    ]
    if not all(path.exists() for path in required):
        return None, None, None
    return load_data()


# ── helpers ───────────────────────────────────────────────────────────────────
def stars(rating):
    rating = float(rating or 0)
    full = int(round(rating))
    full = max(0, min(full, 5))
    return "★" * full + "☆" * (5 - full)
import ast


def format_tags(tags_value):
    """
    Convert tags into a readable comma-separated string.

    Handles:
    - Python lists
    - stringified Python lists
    - plain strings
    """
    if tags_value is None:
        return ""

    if isinstance(tags_value, list):
        return ", ".join(str(x) for x in tags_value if str(x).strip())

    if isinstance(tags_value, str):
        raw = tags_value.strip()
        if not raw:
            return ""

        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed if str(x).strip())
        except Exception:
            pass

        return raw

    return str(tags_value)


def get_modal_row():
    """
    Return the currently selected row for the modal.
    """
    modal_book_id = st.session_state.get("modal_book_id", None)
    results = st.session_state.get("thematic_results", None)

    if modal_book_id is None or results is None or results.empty:
        return None

    selected = results[results["book_id"].astype(str) == str(modal_book_id)]
    if selected.empty:
        return None

    return selected.iloc[0]

def has_real_value(value):
    """
    Return True only when the value is meaningful for display.

    Hide values such as:
    - None
    - empty string
    - 0
    - "0"
    - NaN
    """
    if value is None:
        return False

    if pd.isna(value):
        return False

    # Hide numeric zero
    if isinstance(value, (int, float)) and value == 0:
        return False

    # Hide empty / zero-like strings
    text = str(value).strip()
    if text == "" or text == "0":
        return False

    return True


def book_card(row, source="search"):
    """
    Render a book card with:
    - left cover image
    - center content styled like the recommendation page
    - right add button
    - clickable title that opens the detail modal
    """

    bid = str(row["book_id"])
    title = str(row.get("title", "Unknown Title") or "Unknown Title")

    avg = float(row.get("average_rating") or 0)
    ratings_count = int(row.get("ratings_count") or 0)
    
    # Decide the score to show based on different sources (search relevance vs recommendation similarity)
    metric_text = ""

    if source == "search":
        score = float(row.get("score") or 0)

        # Convert 0-1 → 0-100 if needed
        if score <= 1:
            score *= 100

        metric_text = f"Relevance {score:.0f}"

    elif source == "rec":
        sim = float(row.get("similarity") or 0)
        metric_text = f"Match {sim:.0%}"

    desc = str(row.get("description", "") or "")
    desc_preview = desc[:220] + "…" if len(desc) > 220 else desc

    image_url = str(row.get("image_url", "") or "").strip()

    # Layout: image | styled content card | add button
    col_img, col_main, col_actions = st.columns([1.0, 5.2, 1.2], gap="medium")

    # -------------------------
    # Left: cover image
    # -------------------------
    with col_img:
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="
                    width:100%;
                    height:150px;
                    border:1px solid #e0d5c5;
                    border-radius:8px;
                    background:#f4eee6;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:#9a8878;
                    font-size:0.9rem;
                    text-align:center;
                ">
                    No cover
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------
    # Middle: title, metadata, description
    # -------------------------
    with col_main:
        with col_main:
            st.markdown(
                f"""
                <div class="book-card">
                    <div class="title">{title}</div>
                    <div class="meta">
                        <span class="stars">{stars(avg)}</span>
                        &nbsp;{avg:.2f} · {ratings_count:,} ratings
                        &nbsp;·&nbsp;<b style="color:#b87333;">{metric_text}</b>
                    </div>
                    {f'<div class="desc">{desc_preview}</div>' if desc_preview else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )



    # -------------------------
    # Right: add button
    # -------------------------
    with col_actions:
        st.markdown("<div style='height:1.55rem;'></div>", unsafe_allow_html=True)
        
        if st.button("Open", key=f"title_{bid}_open"):
                st.session_state.modal_book_id = bid
                st.session_state.show_modal = True
                st.rerun()
                
        if bid not in st.session_state.liked_books:
            if st.button("+ Add", key=f"detail_add_{bid}"):
                st.session_state.liked_books[bid] = title
                refresh_recs(rec_embeddings, rec_book_ids, rec_books_df)
                st.rerun()
        else:
            st.markdown("✓ Added")
                
@st.dialog("Book Details", width="large")
def show_modal():
    """
    Show the detail modal for the current page.
    Search page uses thematic_results.
    Recommendation page uses recs.
    """

    bid = st.session_state.modal_book_id

    # Choose the correct dataframe based on the active page
    if st.session_state.active_tab == "📚 Recommendations":
        results = st.session_state.recs
    else:
        results = st.session_state.thematic_results

    # If the source dataframe is missing, reset modal state and stop
    if results is None or results.empty:
        st.session_state.show_modal = False
        st.session_state.modal_book_id = None
        st.rerun()

    row = results[results["book_id"].astype(str) == str(bid)]

    # If the row is not found, reset modal state instead of showing a stale popup
    if row.empty:
        st.session_state.show_modal = False
        st.session_state.modal_book_id = None
        st.rerun()

    row = row.iloc[0]

    # Extract fields
    title = row.get("title", "")
    desc = row.get("description", "")
    avg = float(row.get("average_rating") or 0)
    ratings_count = int(row.get("ratings_count") or 0)
    score = float(row.get("score") or 0) * 100
    image_url = row.get("image_url", "")
    publication_year = row.get("publication_year")
    num_pages = row.get("num_pages")
    publisher = row.get("publisher")
    language_code = row.get("language_code")
    top_tags = format_tags(row.get("top_tags"))
    overall_judgment = str(row.get("overall_judgment", "") or "").strip()

    # Layout inside modal
    col1, col2 = st.columns([1, 3])

    # Left: cover image
    with col1:
        if image_url:
            st.image(image_url)
            
        if bid not in st.session_state.liked_books:
            if st.button("+ Add", key=f"detail_add_{bid}_modal"):
                st.session_state.liked_books[bid] = title
                refresh_recs(rec_embeddings, rec_book_ids, rec_books_df)
                st.rerun()
        else:
            st.markdown("✓ Added")
        

    # Right: metadata + description
    with col2:
        st.markdown(f"## {title}")

        st.write(f"{stars(avg)} {avg:.2f} · {ratings_count:,} ratings")
        st.write(f"Relevance: {score:.0f}/100")
        
        st.write("---")
        st.write(desc)

        # Show only fields that have meaningful values
        if has_real_value(publication_year):
            st.markdown(f"**Publication year:** {publication_year}")

        if has_real_value(num_pages):
            st.markdown(f"**Pages:** {num_pages}")

        if has_real_value(publisher):
            st.markdown(f"**Publisher:** {publisher}")

        if has_real_value(language_code):
            st.markdown(f"**Language code:** {language_code}")

        if has_real_value(top_tags):
            st.markdown(f"**Top tags:** {top_tags}")

        if has_real_value(overall_judgment):
            st.markdown(f"**Overall judgment:** {overall_judgment}")
            
        


    # Close button
    if st.button("Close"):
        st.session_state.show_modal = False
        st.session_state.modal_book_id = None
        st.rerun()
        
    
# ── recommendations ────────────────────────────────────────────────────────────
def refresh_recs(rec_embeddings, rec_book_ids, rec_books_df):
    """
    Refresh recommendation results, then enrich them with the same
    UI/detail fields used by search cards and the modal.
    """
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
        recs = recommend(
            liked_ids,
            rec_embeddings,
            rec_book_ids,
            rec_books_df,
            top_n=5,
        )
        st.session_state.recs_mode = "personalized"

    # Add the same detail fields used by search cards/modal
    detail_cols = [
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

    detail_lookup = books[detail_cols].drop_duplicates("book_id").copy()

    recs["book_id"] = recs["book_id"].astype(str)

    recs = recs.merge(
        detail_lookup,
        on="book_id",
        how="left",
        sort=False,
        suffixes=("", "_ui"),
    )

    # Convert similarity into the same display score used by search cards
    if "similarity" in recs.columns:
        recs["score"] = pd.to_numeric(recs["similarity"], errors="coerce") * 100

    st.session_state.recs = recs


# ── controversy scoring ────────────────────────────────────────────────────────
# Delegated to src.controversy — see that module for implementation details.

def make_book_label(row):
    year = row.get("publication_year", "")
    year_str = ""
    if pd.notna(year) and str(year).strip() != "":
        try:
            year_str = f" ({int(float(year))})"
        except Exception:
            year_str = f" ({year})"

    ratings = int(row.get("ratings_count", 0) or 0)
    return f"{row['title']}{year_str} — {ratings:,} ratings"


def init_session_state():
    defaults = {
        "active_tab": "🔍 Thematic Search",
        "search_page": 1,
        "results_per_page": 10,
        "modal_book_id": None,      # which book is currently shown in modal
        "show_modal": False,        # whether the modal is currently open

        # thematic search state
        "thematic_query": "",
        "thematic_results": None,
        "thematic_searched": False,

        # recommendations search state
        "liked_books": {},
        "recs": pd.DataFrame(),
        "recs_mode": "top_rated",

        # controversy state
        "cq": "",
        "controversy_hits": None,
        "controversy_selected_label": None,
        "controversy_searched": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── UI ─────────────────────────────────────────────────────────────────────────
init_session_state()

st.markdown("""
<div class="hero">
    <h1><span class="radar-dot"></span>ReadRadar</h1>
    <div class="subtitle">Thematic Literature Explorer · UCSD Book Graph Prototype</div>
</div>
""", unsafe_allow_html=True)

books, rec_embeddings, rec_book_ids, rec_books_df, full_books, universe_df = safe_load()

# Initialize recommendation results once after data is loaded
if st.session_state.recs.empty:
    refresh_recs(rec_embeddings, rec_book_ids, rec_books_df)

if books is None:
    st.error(
        "**Data not found.** Place `proto_books.parquet`, `proto_interactions.parquet`, "
        "and `proto_reviews.parquet` in `./data/proto/` then restart."
    )
    st.stop()

# ── persistent top navigation ────────────────────────────────────────────────
nav_options = ["🔍 Thematic Search", "📚 Recommendations", "⚡ Controversy"]

# Initialize the radio widget state only once
if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = st.session_state.active_tab

selected_tab = st.radio(
    "Navigation",
    nav_options,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_tab",
)

# Keep active_tab in sync with the widget value
st.session_state.active_tab = selected_tab

# PAGE 1 — THEMATIC SEARCH
if st.session_state.active_tab == "🔍 Thematic Search":
    st.markdown('<div class="section-label">Describe what you\'re looking for</div>', unsafe_allow_html=True)

    with st.form("thematic_search_form", clear_on_submit=False):
        col_q, col_btn = st.columns([5, 1], vertical_alignment="bottom")
        with col_q:
            query = st.text_input(
                "query",
                placeholder="e.g. dark academia, found family, enemies to lovers in a Victorian setting…",
                label_visibility="collapsed",
                key="thematic_query",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.form_submit_button("Search")

    if search_btn:
        st.session_state.thematic_searched = True
        st.session_state.search_page = 1

        if query.strip():
            with st.spinner("Searching..."):
                raw_results = search.thematic_search(query.strip(), top_k=100)

                if raw_results is None or raw_results.empty:
                    st.session_state.thematic_results = raw_results
                else:
                    raw_results["book_id"] = raw_results["book_id"].astype(str)

                    detail_cols = [
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
                    detail_lookup = books[detail_cols].drop_duplicates("book_id")

                    st.session_state.thematic_results = raw_results.merge(
                        detail_lookup,
                        on="book_id",
                        how="left",
                        sort=False,
                        suffixes=("", "_ui"),
                    )
        else:
            st.session_state.thematic_results = None
            
    # =========================
    # SEARCH RESULTS LOGIC
    # =========================

    if st.session_state.thematic_searched:

        # --- Case 1: no query ---
        if not st.session_state.thematic_query.strip():
            st.warning("Please enter a query.")

        # --- Case 2: no results ---
        elif st.session_state.thematic_results is None or st.session_state.thematic_results.empty:
            st.info("No matches found.")

        else:
            results = st.session_state.thematic_results


            # =========================
            # LIST VIEW MODE
            # =========================
            total = len(results)
            per_page = st.session_state.results_per_page

            # calculate pagination
            total_pages = (total + per_page - 1) // per_page
            page = st.session_state.search_page

            start = (page - 1) * per_page
            end = start + per_page

            page_results = results.iloc[start:end]

            st.write(f"{total} results")

            # --- render cards ---
            for _, row in page_results.iterrows():
                book_card(row, source="search")

            # --- open modal if needed ---
            if st.session_state.show_modal:
                show_modal()

            # --- pagination controls ---
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if page > 1:
                    if st.button("Previous", key="search_prev_page"):
                        st.session_state.search_page -= 1
                        st.rerun()

            with col2:
                st.markdown(f"Page {page} / {total_pages}")

            with col3:
                if page < total_pages:
                    if st.button("Next", key="search_next_page"):
                        st.session_state.search_page += 1
                        st.rerun()

                        
    with st.expander("💡 Search tips"):
        st.markdown("""
- **Themes**: `redemption arc`, `morally grey protagonist`, `slow burn romance`
- **Settings**: `post-apocalyptic`, `Victorian London`, `high fantasy court`
- **Moods**: `cozy mystery`, `dark and atmospheric`, `hopeful`
- **Subjects**: `time travel`, `dragons`, `chosen one`, `magic academy`
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "📚 Recommendations":
    st.markdown('<div class="list-section-title">Your Reading List</div>', unsafe_allow_html=True)

    liked_items = list(st.session_state.liked_books.items())

    if liked_items:
        for bid, title in liked_items:
            row_data = rec_books_df[rec_books_df["book_id"] == bid]
            avg = float(row_data["average_rating"].values[0]) if not row_data.empty else 0.0

            col_info, col_btn = st.columns([8, 1])

            with col_info:
                st.markdown(
                    f"<div style='padding:0.5rem 0;border-bottom:1px solid #f0ebe3;'>"
                    f"<div class='bc-title-sm'>{title}</div>"
                    f"<div class='bc-meta-sm'><span class='bc-stars'>{stars(avg)}</span> {avg:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_btn:
                st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"rm_{bid}"):
                    del st.session_state.liked_books[bid]
                    refresh_recs(rec_embeddings, rec_book_ids, rec_books_df)
                    st.rerun()
    else:
        st.info("Your reading list is empty. Add books from search results.")

    st.markdown('<div class="rec-title">Recommended for You</div>', unsafe_allow_html=True)

    desc_lookup = full_books.set_index("book_id")["description"].to_dict()

    for _, row in st.session_state.recs.iterrows():
        book_card(row, source="rec")

    if st.session_state.show_modal:
        show_modal()
    
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CONTROVERSY
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "⚡ Controversy":
    st.markdown('<div class="section-label">Search for a book or author</div>', unsafe_allow_html=True)

    with st.form("controversy_search_form", clear_on_submit=False):
        col_cq, col_cbtn = st.columns([5, 1])
        with col_cq:
            c_query = st.text_input(
                "controversy_query",
                placeholder="e.g. Twilight, Throne of Glass, George R.R. Martin…",
                label_visibility="collapsed",
                key="cq",
            )
        with col_cbtn:
            st.markdown("<br>", unsafe_allow_html=True)
            c_btn = st.form_submit_button("Analyse")

    if c_btn:
        st.session_state.controversy_searched = True
        if c_query.strip():
            with st.spinner("Searching…"):
                hits = search.thematic_search(c_query.strip(), books, top_k=8)
            if hits.empty:
                st.session_state.controversy_hits = None
                st.session_state.controversy_selected_label = None
            else:
                hits = hits.copy()
                hits["label"] = hits.apply(make_book_label, axis=1)
                st.session_state.controversy_hits = hits
                st.session_state.controversy_selected_label = hits.iloc[0]["label"]
        else:
            st.session_state.controversy_hits = None
            st.session_state.controversy_selected_label = None

    if st.session_state.controversy_searched:
        hits = st.session_state.controversy_hits

        if not st.session_state.cq.strip():
            st.warning("Please enter a book title or author.")
        elif hits is None or hits.empty:
            st.info("No books found.")
        else:
            labels = hits["label"].tolist()

            # preserve selected option across reruns and allow changing it
            if st.session_state.controversy_selected_label not in labels:
                st.session_state.controversy_selected_label = labels[0]

            selected_label = st.selectbox(
                "Select a book:",
                labels,
                index=labels.index(st.session_state.controversy_selected_label),
                key="controversy_picker",
            )

            st.session_state.controversy_selected_label = selected_label
            chosen_row = hits[hits["label"] == selected_label].iloc[0]
            chosen_id = str(chosen_row["book_id"])

            score, mean_rating, rating_dist = controversy.compute_controversy(chosen_id, interactions)
            pros, cons = controversy.extract_pros_cons(chosen_id, reviews)

            st.markdown("---")

            # ── controversy meter ──
            if score is not None:
                pct = score
                label = "Polarising" if pct >= 70 else ("Mixed" if pct >= 40 else "Generally loved")
                gradient = (
                    "linear-gradient(90deg,#7c1e1e,#c86e6e)"
                    if pct >= 70
                    else "linear-gradient(90deg,#6e4c1e,#c8a96e)"
                )

                st.markdown(f"""
                <div class="controversy-meter">
                    <div class="score-number">{pct:.0f}</div>
                    <div class="score-label">Controversy Score · {label}</div>
                    <div class="bar-bg"><div class="bar-fill" style="width:{pct}%; background: {gradient}"></div></div>
                </div>
                """, unsafe_allow_html=True)

                if rating_dist is not None and len(rating_dist) > 0:
                    col_a, col_b = st.columns([2, 3])
                    with col_a:
                        st.markdown('<div class="section-label">Rating distribution</div>', unsafe_allow_html=True)
                        dist_df = rating_dist.reset_index()
                        dist_df.columns = ["Rating", "Count"]
                        dist_df["Rating"] = dist_df["Rating"].astype(str) + " ★"
                        st.bar_chart(dist_df.set_index("Rating"), height=160)

                    with col_b:
                        st.markdown(f"""
                        <div class="section-label">Book overview</div>
                        <div class="book-card">
                            <div class="title">{chosen_row['title']}</div>
                            <div class="meta">Avg rating: {mean_rating:.2f} · {int(chosen_row.get('ratings_count', 0)):,} ratings</div>
                            <div class="desc">{str(chosen_row.get('description', ''))[:300]}…</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Not enough ratings data for a controversy score.")

            # ── pros & cons ──
            if pros or cons:
                col_p, col_c = st.columns(2)
                with col_p:
                    st.markdown('<div class="section-label">✦ What readers love</div>', unsafe_allow_html=True)
                    if pros:
                        for p in pros:
                            st.markdown(f'<div class="pro-item">"{p}"</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="pro-item">No positive highlights found in sample reviews.</div>', unsafe_allow_html=True)

                with col_c:
                    st.markdown('<div class="section-label">✦ Common criticisms</div>', unsafe_allow_html=True)
                    if cons:
                        for c in cons:
                            st.markdown(f'<div class="con-item">"{c}"</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="con-item">No critical highlights found in sample reviews.</div>', unsafe_allow_html=True)
            else:
                st.info("No review text found for this book in the current sample.")


# ── data credit ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="data-credit">
    Data: <a href="https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home" target="_blank">
    UCSD Book Graph</a> (Goodreads snapshot, 2017) · Wan et al., 2018 ·
    Metadata enrichment: Open Library + Wikipedia
</div>
""", unsafe_allow_html=True)