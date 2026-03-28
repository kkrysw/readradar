"""
ReadRadar — Thematic Literature Explorer
Streamlit prototype using UCSD Book Graph (fantasy/paranormal subset)
Data source: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

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
    margin-top: 0.5rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.hero .radar-dot {
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
</style>
""", unsafe_allow_html=True)


# ── data loading ───────────────────────────────────────────────────────────────
DATA_DIR = Path("data/proto")


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
    books = pd.read_parquet(DATA_DIR / "proto_books.parquet")
    interactions = pd.read_parquet(DATA_DIR / "proto_interactions.parquet")
    reviews = pd.read_parquet(DATA_DIR / "proto_reviews.parquet")

    # clean up scalar types
    books["ratings_count"] = pd.to_numeric(books["ratings_count"], errors="coerce").fillna(0).astype(int)
    books["average_rating"] = pd.to_numeric(books["average_rating"], errors="coerce")
    books["book_id"] = books["book_id"].astype(str)
    interactions["book_id"] = interactions["book_id"].astype(str)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    reviews["book_id"] = reviews["book_id"].astype(str)

    # sanitize columns that store lists-of-dicts — parquet deserialises these as
    # numpy arrays of dicts, which Streamlit's DataFrame hasher can't hash
    for col in ("popular_shelves", "authors"):
        if col in books.columns:
            books[col] = _sanitize_object_col(books[col])

    return books, interactions, reviews


def safe_load():
    if not (DATA_DIR / "proto_books.parquet").exists():
        return None, None, None
    return load_data()


# ── helpers ───────────────────────────────────────────────────────────────────
def stars(rating):
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def book_card(row, show_score=None, score_label=None):
    desc = str(row.get("description", ""))
    desc_preview = desc[:200] + "…" if len(desc) > 200 else desc
    avg = row.get("average_rating", 0) or 0
    pill = ""
    if show_score is not None:
        pill = f'<div class="score-pill">{score_label}: {show_score:.0f}/100</div>'
    st.markdown(f"""
    <div class="book-card">
        <div class="title">{row['title']}</div>
        <div class="meta">
            <span class="stars">{stars(avg)}</span>
            &nbsp;{avg:.2f} · {int(row.get('ratings_count', 0)):,} ratings
        </div>
        <div class="desc">{desc_preview}</div>
        {pill}
    </div>
    """, unsafe_allow_html=True)


# ── thematic search ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    books, _, _ = load_data()  # safe: load_data is cache_data, already sanitized

    def shelf_text(shelves):
        # guard against None, NaN, scalars, and numpy arrays
        if shelves is None:
            return ""
        try:
            items = list(shelves)  # works for lists, np arrays, and other iterables
        except TypeError:
            return ""
        if not items:
            return ""
        names = []
        for s in items:
            if isinstance(s, dict):
                names.append(s.get("name", "").replace("-", " "))
        return " ".join(names)

    corpus = (
        books["description"].fillna("") + " " +
        books["title"].fillna("") + " " +
        books["popular_shelves"].apply(shelf_text)
    )
    vec = TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2))
    mat = vec.fit_transform(corpus)
    return vec, mat


def thematic_search(query, books, vec, mat, top_k=10):
    from sklearn.metrics.pairwise import cosine_similarity
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, mat).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = books.iloc[top_idx].copy()
    results["_score"] = sims[top_idx]
    return results[results["_score"] > 0]


# ── recommendations ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rec_model():
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    _, interactions, _ = load_data()  # safe: already sanitized by load_data

    # encode ids
    users = interactions["user_id"].astype("category")
    books_cat = interactions["book_id"].astype("category")
    ratings = interactions["rating"].fillna(0)

    mat = csr_matrix(
        (ratings.values, (users.cat.codes.values, books_cat.cat.codes.values))
    )
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(mat)
    item_factors = svd.components_.T  # shape: (n_books, n_components)

    return svd, item_factors, books_cat.cat.categories.tolist()


def recommend_from_books(liked_ids, books_df, interactions, top_k=8):
    try:
        svd, item_factors, book_cats = build_rec_model()
    except Exception:
        # fallback: popularity-based
        return books_df[~books_df["book_id"].isin(liked_ids)].nlargest(top_k, "ratings_count")

    # build pseudo-user vector by averaging item factors of liked books
    idxs = [book_cats.index(bid) for bid in liked_ids if bid in book_cats]
    if not idxs:
        return books_df[~books_df["book_id"].isin(liked_ids)].nlargest(top_k, "ratings_count")

    pseudo_user = item_factors[idxs].mean(axis=0)
    scores = item_factors @ pseudo_user
    top_idx = np.argsort(scores)[::-1]
    ranked_book_ids = [book_cats[i] for i in top_idx]
    ranked_book_ids = [b for b in ranked_book_ids if b not in liked_ids][:top_k * 3]

    result = books_df[books_df["book_id"].isin(ranked_book_ids)].copy()
    id_to_score = {b: scores[book_cats.index(b)] for b in ranked_book_ids if b in book_cats}
    result["_rec_score"] = result["book_id"].map(id_to_score)
    return result.nlargest(top_k, "_rec_score")


# ── controversy scoring ────────────────────────────────────────────────────────
def compute_controversy(book_id, interactions):
    rows = interactions[interactions["book_id"] == book_id]["rating"].dropna()
    if len(rows) < 5:
        return None, None, None
    mean_ = rows.mean()
    std_ = rows.std()
    count_ = len(rows)
    # higher std + more ratings → more controversial
    raw = std_ * np.log1p(count_)
    # normalise to 0-100 using empirical max ~3.0
    score = min(100, raw / 3.0 * 100)
    return score, mean_, rows.value_counts().sort_index()


def extract_pros_cons(book_id, reviews, n=4):
    """Simple sentiment split using polarity word lists."""
    pos_words = {
        "love", "loved", "amazing", "beautiful", "wonderful", "brilliant",
        "fantastic", "incredible", "perfect", "excellent", "great", "best",
        "masterpiece", "compelling", "captivating", "engaging", "stunning",
        "magical", "powerful", "moving", "enchanting", "riveting", "outstanding",
    }
    neg_words = {
        "hate", "hated", "boring", "awful", "terrible", "worst", "disappointing",
        "bad", "poor", "weak", "slow", "dull", "predictable", "annoying",
        "frustrating", "confusing", "overrated", "tedious", "ridiculous", "cheesy",
        "stupid", "bland", "mediocre", "cliché",
    }

    book_reviews = reviews[reviews["book_id"] == book_id]["review_text"].dropna()
    pros, cons = [], []

    for review in book_reviews:
        sentences = [s.strip() for s in str(review).replace("!", ".").split(".") if len(s.strip()) > 30]
        for sent in sentences:
            lower = sent.lower()
            words = set(lower.split())
            pos_hits = len(words & pos_words)
            neg_hits = len(words & neg_words)
            if pos_hits > neg_hits and len(pros) < n * 3:
                pros.append((pos_hits, sent[:180]))
            elif neg_hits > pos_hits and len(cons) < n * 3:
                cons.append((neg_hits, sent[:180]))

    pros = [s for _, s in sorted(pros, reverse=True)[:n]]
    cons = [s for _, s in sorted(cons, reverse=True)[:n]]
    return pros, cons

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
        "liked_books": {},

        # thematic search state
        "thematic_query": "",
        "thematic_results": None,
        "thematic_searched": False,

        # recommendations search state
        "liked_search": "",

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

books, interactions, reviews = safe_load()

if books is None:
    st.error(
        "**Data not found.** Place `proto_books.parquet`, `proto_interactions.parquet`, "
        "and `proto_reviews.parquet` in `./data/proto/` then restart."
    )
    st.stop()

with st.spinner("Building search index…"):
    vec, mat = build_tfidf()

# ── persistent top navigation ────────────────────────────────────────────────
nav_options = ["🔍 Thematic Search", "📚 Recommendations", "⚡ Controversy"]
current_idx = nav_options.index(st.session_state.active_tab) if st.session_state.active_tab in nav_options else 0

st.session_state.active_tab = st.radio(
    "Navigation",
    nav_options,
    index=current_idx,
    horizontal=True,
    label_visibility="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — THEMATIC SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "🔍 Thematic Search":
    st.markdown('<div class="section-label">Describe what you\'re looking for</div>', unsafe_allow_html=True)

    with st.form("thematic_search_form", clear_on_submit=False):
        col_q, col_btn = st.columns([5, 1])
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
        if query.strip():
            with st.spinner("Searching…"):
                st.session_state.thematic_results = thematic_search(query.strip(), books, vec, mat)
        else:
            st.session_state.thematic_results = None

    if st.session_state.thematic_searched:
        if not st.session_state.thematic_query.strip():
            st.warning("Please enter a theme, subject, or title to search.")
        elif st.session_state.thematic_results is None or st.session_state.thematic_results.empty:
            st.info("No matches found. Try different keywords.")
        else:
            results = st.session_state.thematic_results
            st.markdown(
                f'<div class="section-label">{len(results)} results for "{st.session_state.thematic_query}"</div>',
                unsafe_allow_html=True,
            )
            for _, row in results.iterrows():
                book_card(row, show_score=row["_score"] * 100, score_label="Relevance")

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
    st.markdown('<div class="section-label">Find books you\'ve read</div>', unsafe_allow_html=True)

    search_liked = st.text_input(
        "liked_search",
        placeholder="Search for a book you liked…",
        label_visibility="collapsed",
        key="liked_search",
    )

    if search_liked.strip():
        hits = thematic_search(search_liked.strip(), books, vec, mat, top_k=5)
        for _, row in hits.iterrows():
            bid = str(row["book_id"])
            cols = st.columns([6, 1])
            with cols[0]:
                st.markdown(f"**{row['title']}** — ⭐ {row['average_rating']:.2f}")
            with cols[1]:
                if bid not in st.session_state.liked_books:
                    if st.button("+ Add", key=f"add_{bid}"):
                        st.session_state.liked_books[bid] = row["title"]
                        st.rerun()
                else:
                    st.markdown("✓ Added")

    if st.session_state.liked_books:
        st.markdown('<div class="section-label">Your liked books</div>', unsafe_allow_html=True)
        for bid, title in list(st.session_state.liked_books.items()):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"📖 {title}")
            if c2.button("✕", key=f"rm_{bid}"):
                del st.session_state.liked_books[bid]
                st.rerun()

        if st.button("Get Recommendations →", key="get_recs"):
            with st.spinner("Finding books you'll love…"):
                liked_ids = list(st.session_state.liked_books.keys())
                recs = recommend_from_books(liked_ids, books, interactions)

            st.markdown('<div class="section-label">Recommended for you</div>', unsafe_allow_html=True)
            for _, row in recs.iterrows():
                book_card(row)
    else:
        st.info("Add at least one book you've liked to get recommendations.")

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
                hits = thematic_search(c_query.strip(), books, vec, mat, top_k=8)
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

            score, mean_rating, rating_dist = compute_controversy(chosen_id, interactions)
            pros, cons = extract_pros_cons(chosen_id, reviews)

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