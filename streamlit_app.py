
"""
streamlit_app.py

Minimal Streamlit UI for the SQLite catalog produced by ingest_catalog.py

Run:
  streamlit run streamlit_app.py

Expected DB:
  data/catalog.db  (or set CATALOG_DB env var)
"""

import os
import json
import sqlite3
import pandas as pd
import streamlit as st


DB_PATH = os.getenv("CATALOG_DB", os.path.join("data", "catalog.db"))

@st.cache_data(show_spinner=False)
def load_catalog(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    # Join works + authors (1 author per work in our seed; keep it simple)
    q = """
    SELECT
      w.work_id,
      w.title,
      w.pub_year,
      w.description,
      w.genre_wikidata,
      w.gutenberg_id,
      w.ol_work_key,
      w.wikidata_entity,
      w.sg_url,
      a.name AS author
    FROM works w
    LEFT JOIN work_authors wa ON wa.work_id = w.work_id
    LEFT JOIN authors a ON a.author_id = wa.author_id
    """
    df = pd.read_sql_query(q, con)
    con.close()
    return df

@st.cache_data(show_spinner=False)
def load_subjects_tags(db_path: str):
    con = sqlite3.connect(db_path)
    subj = pd.read_sql_query("""
        SELECT ws.work_id, s.subject
        FROM work_subjects ws
        JOIN subjects s ON s.subject_id = ws.subject_id
    """, con)
    tags = pd.read_sql_query("""
        SELECT wt.work_id, t.tag
        FROM work_tags wt
        JOIN tags t ON t.tag_id = wt.tag_id
    """, con)
    con.close()
    return subj, tags

def contains_any(text: str, needles):
    t = (text or "").lower()
    for n in needles:
        if n and n.lower() in t:
            return True
    return False


st.set_page_config(page_title="Thematic Literature Map", layout="wide")
st.title("Thematic Literature Map — Catalog Explorer")

if not os.path.exists(DB_PATH):
    st.error(f"Catalog DB not found at: {DB_PATH}\n\nRun: python ingest_catalog.py --out data/catalog.db")
    st.stop()

df = load_catalog(DB_PATH)
subj, tags = load_subjects_tags(DB_PATH)

# Build per-work aggregates for fast filtering
subj_agg = subj.groupby("work_id")["subject"].apply(list).to_dict()
tags_agg = tags.groupby("work_id")["tag"].apply(list).to_dict()

df["subjects"] = df["work_id"].map(lambda x: subj_agg.get(x, []))
df["tags"] = df["work_id"].map(lambda x: tags_agg.get(x, []))

with st.sidebar:
    st.header("Search")
    query = st.text_input("Theme / subject / title / author", value="")
    year_min, year_max = st.slider("Publication year range (rough)", 1500, 2026, (1800, 2026))
    require_storygraph = st.checkbox("Only show books with StoryGraph page", value=False)
    max_rows = st.selectbox("Max results", [25, 50, 100, 200], index=1)

# Filter
df2 = df.copy()

# year filter (best effort: pub_year may be empty or non-numeric)
def year_ok(y):
    try:
        yi = int(str(y)[:4])
        return year_min <= yi <= year_max
    except Exception:
        return True  # keep unknowns

df2 = df2[df2["pub_year"].apply(year_ok)]

if require_storygraph:
    df2 = df2[df2["sg_url"].fillna("").str.len() > 0]

q = query.strip().lower()
if q:
    def match_row(r):
        if q in (r["title"] or "").lower():
            return True
        if q in (r["author"] or "").lower():
            return True
        # subjects/tags match
        if contains_any(" ".join(r["subjects"]), [q]):
            return True
        if contains_any(" ".join(r["tags"]), [q]):
            return True
        # description (light)
        if q in (r["description"] or "").lower():
            return True
        return False

    df2 = df2[df2.apply(match_row, axis=1)]

df2 = df2.sort_values(["pub_year", "title"], ascending=[False, True]).head(int(max_rows))

st.caption(f"Loaded {len(df):,} works. Showing {len(df2):,} matches from {os.path.basename(DB_PATH)}.")

# Display
def fmt_list(xs, n=10):
    xs = xs or []
    xs = [x for x in xs if x]
    if len(xs) > n:
        return ", ".join(xs[:n]) + f" … (+{len(xs)-n})"
    return ", ".join(xs)

# A clean table + a details expander
cols = ["title", "author", "pub_year", "genre_wikidata", "gutenberg_id", "ol_work_key", "sg_url"]
st.dataframe(
    df2[cols],
    use_container_width=True,
    hide_index=True
)

st.subheader("Details")
for _, r in df2.iterrows():
    label = f"{r['title']} — {r.get('author','') or 'Unknown'} ({r.get('pub_year','') or 'year ?'})"
    with st.expander(label):
        st.write(r.get("description","") or "")
        st.markdown(f"**Subjects:** {fmt_list(r.get('subjects',[]), n=18)}")
        st.markdown(f"**StoryGraph tags:** {fmt_list(r.get('tags',[]), n=18)}")
        if r.get("sg_url"):
            st.markdown(f"**StoryGraph:** {r['sg_url']}")
