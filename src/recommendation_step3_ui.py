"""
recommendation_step3_ui.py — ReadRadar Recommendation UI
Streamlit prototype for the book recommendation feature.

Layout:
  1. Header
  2. Reading List   — Inline book rows with title+rating+✕, no box wrapper
  3. Recommendations — 5 books auto-refreshed; top-rated if list empty; + Add button
  4. 3D Universe    — Interactive Plotly scatter; liked books highlighted in orange

Run:
  streamlit run src/recommendation_step3_ui.py

Dependencies:
  pip install streamlit pandas pyarrow plotly numpy scikit-learn
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from recommendation_step2_algorithm import load_artifacts, recommend

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ReadRadar · For You",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,800;1,400&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #f7f3ee !important;
    color: #2c2420 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background-color: #f7f3ee; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1080px; }

/* hero */
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
    margin-top: 0.5rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.radar-dot {
    display: inline-block;
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

/* section titles */
.list-section-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1210;
    text-align: center;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}
.rec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a1210;
    margin-bottom: 1.2rem;
}

/* book card (recommendations) */
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
.bc-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a1210;
    margin-bottom: 0.2rem;
}
.bc-title-sm {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a1210;
    margin-bottom: 0.1rem;
}
.bc-meta {
    font-size: 0.78rem;
    color: #9a8878;
    margin-bottom: 0.5rem;
}
.bc-meta-sm {
    font-size: 0.75rem;
    color: #9a8878;
}
.bc-desc {
    font-size: 0.875rem;
    color: #4a3c30;
    line-height: 1.6;
}
.bc-stars { color: #c8920a; }

/* buttons */
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
    padding: 0.5rem 1.1rem !important;
    transition: background 0.2s, transform 0.1s !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    background: #9e6228 !important;
    transform: translateY(-1px) !important;
}

/* data credit */
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
.data-credit a:hover { color: #b87333; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("data/artifacts")
PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all():
    factors_df, books_df = load_artifacts()
    books_df["book_id"] = books_df["book_id"].astype(str)
    books_df["average_rating"] = pd.to_numeric(books_df["average_rating"], errors="coerce")
    books_df["ratings_count"] = (
        pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0).astype(int)
    )

    full_books = pd.read_parquet(
        PROCESSED_DIR / "books.parquet",
        columns=["book_id", "title", "description", "average_rating", "ratings_count"],
    )
    full_books["book_id"] = full_books["book_id"].astype(str)
    full_books["average_rating"] = pd.to_numeric(full_books["average_rating"], errors="coerce")
    full_books["ratings_count"] = (
        pd.to_numeric(full_books["ratings_count"], errors="coerce").fillna(0).astype(int)
    )

    universe_df = pd.read_parquet(ARTIFACTS_DIR / "read_universe_3d.parquet")
    universe_df["book_id"] = universe_df["book_id"].astype(str)
    rating_meta = books_df[["book_id", "average_rating", "ratings_count"]].copy()
    universe_df = universe_df.merge(rating_meta, on="book_id", how="left")
    universe_df["Rating_Category"] = universe_df["average_rating"].apply(
        lambda r: "⭐ > 4.5"     if r >= 4.5 else (
                  "⭐ 4.0 - 4.4"  if r >= 4.0 else (
                  "⭐ 3.5 - 3.9"  if r >= 3.5 else
                  "⭐ < 3.5"))
    )
    return factors_df, books_df, full_books, universe_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stars(rating):
    full = int(round(float(rating or 0)))
    return "★" * full + "☆" * (5 - full)


def refresh_recs(factors_df, books_df):
    liked_ids = list(st.session_state.liked_books.keys())
    if not liked_ids:
        top5 = (
            books_df[["book_id", "title", "average_rating", "ratings_count"]]
            .sort_values(["average_rating", "ratings_count"], ascending=[False, False])
            .head(5)
            .reset_index(drop=True)
        )
        top5.insert(0, "rank", top5.index + 1)
        top5["similarity"] = None
        st.session_state.recs = top5
        st.session_state.recs_mode = "top_rated"
    else:
        st.session_state.recs = recommend(liked_ids, factors_df, books_df, top_n=5)
        st.session_state.recs_mode = "personalized"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "liked_books" not in st.session_state:
    st.session_state.liked_books = {}

if "recs" not in st.session_state:
    st.session_state.recs = pd.DataFrame()
    st.session_state.recs_mode = "top_rated"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
factors_df, books_df, full_books, universe_df = load_all()

if st.session_state.recs.empty:
    refresh_recs(factors_df, books_df)


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1><span class="radar-dot"></span>ReadRadar</h1>
    <div class="subtitle">Your Personal Reading Universe</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SECTION 1 — READING LIST
# ---------------------------------------------------------------------------
st.markdown('<div class="list-section-title">Your Reading List</div>', unsafe_allow_html=True)

liked_items = list(st.session_state.liked_books.items())

if liked_items:
    for bid, title in liked_items:
        row_data = books_df[books_df["book_id"] == bid]
        avg = float(row_data["average_rating"].values[0]) if not row_data.empty else 0.0
        col_info, col_btn = st.columns([8, 1])
        with col_info:
            st.markdown(
                f"<div style='padding:0.5rem 0;border-bottom:1px solid #f0ebe3;'>"
                f"<div class='bc-title-sm'>{title}</div>"
                f"<div class='bc-meta-sm'>"
                f"<span class='bc-stars'>{stars(avg)}</span> {avg:.2f}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
            if st.button("✕", key=f"rm_{bid}"):
                del st.session_state.liked_books[bid]
                refresh_recs(factors_df, books_df)
                st.rerun()
else:
    st.markdown(
        "<div style='color:#b0a090;font-style:italic;font-size:0.88rem;"
        "text-align:center;padding:1.5rem 0;'>"
        "Your reading list is empty.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SECTION 2 — RECOMMENDATIONS
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="rec-title">Recommended for You</div>', unsafe_allow_html=True)

desc_lookup = full_books.set_index("book_id")["description"].to_dict()

for _, row in st.session_state.recs.iterrows():
    bid = str(row["book_id"])
    avg = float(row.get("average_rating") or 0)
    sim = row.get("similarity")
    desc = str(desc_lookup.get(bid, "") or "")
    desc_preview = (desc[:240] + "…") if len(desc) > 240 else desc

    meta_extra = (
        f"&nbsp;·&nbsp;Match <b style='color:#b87333;'>{float(sim):.0%}</b>"
        if sim is not None else ""
    )

    col_card, col_btn = st.columns([9, 1])
    with col_card:
        st.markdown(f"""
        <div class="book-card">
            <div class="bc-title">{row['title']}</div>
            <div class="bc-meta">
                <span class="bc-stars">{stars(avg)}</span>
                &nbsp;{avg:.2f}{meta_extra}
            </div>
            <div class="bc-desc">{desc_preview}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)
        if bid in st.session_state.liked_books:
            st.markdown(
                "<div style='color:#b87333;font-size:0.78rem;text-align:center;'>✓ Added</div>",
                unsafe_allow_html=True,
            )
        else:
            if st.button("+ Add", key=f"like_{bid}"):
                st.session_state.liked_books[bid] = str(row["title"])
                refresh_recs(factors_df, books_df)
                st.rerun()


# ---------------------------------------------------------------------------
# SECTION 3 — 3D UNIVERSE
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;margin-top:2rem;margin-bottom:2rem;">
    <div style="margin-bottom:0.5rem;">
        <span style="font-size:3rem;vertical-align:middle;margin-right:10px;">🌌</span>
        <span style="
            font-family:'Playfair Display',serif;
            font-size:3.2rem;
            font-weight:800;
            background:linear-gradient(135deg,#2c2420 0%,#ff5e00 100%);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            letter-spacing:-1px;
            vertical-align:middle;
        ">Explore the 3D Universe</span>
    </div>
    <p style="font-size:1rem;color:#7a6655;font-family:'DM Sans',sans-serif;">
        Every dot is a book. <b style="color:#ff5e00;">Drag to rotate, scroll to zoom.</b><br>
        Books in your reading list light up in orange.
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Rendering 3D universe…"):
    map_df = universe_df.copy()
    liked_ids_set = set(st.session_state.liked_books.keys())

    if liked_ids_set:
        map_df["Is_Liked"] = map_df["book_id"].isin(liked_ids_set)
        map_df["Point_Size"] = map_df["Is_Liked"].map({True: 30, False: 2})
        map_df["Color"] = map_df["Is_Liked"].map({True: "📖 Your List", False: "Other Books"})
        color_col    = "Color"
        color_map    = {"📖 Your List": "#ff5e00", "Other Books": "#2c3e50"}
        cat_orders   = {color_col: ["📖 Your List", "Other Books"]}
        legend_title = "Reading List"
        size_max = 40
        hover_dict = {
            "x": False, "y": False, "z": False,
            "average_rating": True, "ratings_count": True,
            "Rating_Category": False, "Point_Size": False,
            "Is_Liked": False, "Color": False,
        }
    else:
        map_df["Point_Size"] = 3
        color_col    = "Rating_Category"
        color_map    = {
            "⭐ > 4.5":    "#985F49",
            "⭐ 4.0 - 4.4": "#C07653",
            "⭐ 3.5 - 3.9": "#DE9B6B",
            "⭐ < 3.5":    "#EAD2B6",
        }
        cat_orders   = {color_col: ["⭐ > 4.5", "⭐ 4.0 - 4.4", "⭐ 3.5 - 3.9", "⭐ < 3.5"]}
        legend_title = "Book Ratings"
        size_max = 8
        hover_dict = {
            "x": False, "y": False, "z": False,
            "average_rating": True, "ratings_count": True,
            "Rating_Category": False, "Point_Size": False,
        }

    fig = px.scatter_3d(
        map_df,
        x="x", y="y", z="z",
        hover_name="title",
        hover_data=hover_dict,
        color=color_col,
        color_discrete_map=color_map,
        category_orders=cat_orders,
        size="Point_Size",
        size_max=size_max,
        opacity=1.0,
    )
    fig.update_layout(
        font=dict(color="#2c2420"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        legend_title_text=legend_title,
        height=600,
        hoverlabel=dict(
            bgcolor="#463933",
            font_size=14,
            font_color="#f7f3ee",
            bordercolor="rgba(0,0,0,0)",
            font_family="DM Sans, sans-serif",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="data-credit">
    Data: <a href="https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home" target="_blank">
    UCSD Book Graph</a> (Goodreads snapshot, 2017) · Wan et al., 2018
</div>
""", unsafe_allow_html=True)