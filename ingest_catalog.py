
"""
ingest_catalog.py

Build a global book catalog for a Streamlit “Thematic Literature Map” app.

What it does
------------
1) Wikidata SPARQL -> seed list (“canonical” books via Project Gutenberg ID + sitelinks threshold)
2) Open Library -> enrich (work key, description, subjects, covers, ISBNs)
3) Optional StoryGraph scraping -> enrich “vibe” tags (moods/pace-ish tokens visible on public pages)
4) Dedupe -> merge records that refer to the same work
5) Normalize -> write a SQLite database + a flattened CSV

Install
-------
pip install pandas requests beautifulsoup4 rapidfuzz SPARQLWrapper tqdm

Run
---
python ingest_catalog.py --out data/catalog.db --csv data/catalog_flat.csv --limit 300 --min-sitelinks 15 --storygraph 1

Notes / realities
-----------------
- StoryGraph has no official public API for this, so scraping can break any time. Keep it optional.
- Be polite: rate-limit + caching. This script caches HTTP responses under data/cache/.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import argparse
import hashlib
import sqlite3
from typing import Optional, Dict, Any, List, Tuple

import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from SPARQLWrapper import SPARQLWrapper, JSON


# -------------------------
# Config
# -------------------------

DEFAULT_USER_AGENT = "ThematicLiteratureMap/1.0 (contact: your_email@example.com)"
USER_AGENT = os.getenv("TLITMAP_USER_AGENT", DEFAULT_USER_AGENT)

DATA_DIR = os.getenv("TLITMAP_DATA_DIR", "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
OPENLIB_SEARCH = "https://openlibrary.org/search.json"
OPENLIB_WORK_JSON = "https://openlibrary.org{work_key}.json"  # work_key like "/works/OL123W"

STORYGRAPH_BROWSE = "https://app.thestorygraph.com/browse"
STORYGRAPH_BOOK = "https://app.thestorygraph.com/books/{uuid}"

# Conservative throttling
WD_SLEEP = 0.15
OL_SLEEP = 0.15
SG_SLEEP = 0.9

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# -------------------------
# Helpers: caching + http
# -------------------------

def _cache_path(prefix: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{prefix}_{digest}.json")

def cached_get_json(url: str, params: Optional[dict] = None, prefix: str = "http", sleep_s: float = 0.0) -> Optional[dict]:
    key = url + "?" + (json.dumps(params, sort_keys=True) if params else "")
    path = _cache_path(prefix, key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        r = SESSION.get(url, params=params, timeout=35)
        if sleep_s:
            time.sleep(sleep_s)
        if r.status_code != 200:
            return None
        data = r.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data
    except Exception:
        return None

def cached_get_text(url: str, params: Optional[dict] = None, prefix: str = "html", sleep_s: float = 0.0) -> Optional[str]:
    key = url + "?" + (json.dumps(params, sort_keys=True) if params else "")
    path = _cache_path(prefix, key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    try:
        r = SESSION.get(url, params=params, timeout=35)
        if sleep_s:
            time.sleep(sleep_s)
        if r.status_code != 200:
            return None
        text = r.text
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception:
        return None


# -------------------------
# 1) Wikidata seed list
# -------------------------

WIKIDATA_QUERY = """
SELECT DISTINCT ?book ?bookLabel ?authorLabel ?gutenbergId
                ?pubDate ?genreLabel ?sitelinks ?wpTitle
WHERE {
  VALUES ?workType { wd:Q7725634 wd:Q571 wd:Q8261 wd:Q49084 wd:Q5185279 wd:Q25379 }
  ?book wdt:P31 ?workType .
  ?book wdt:P2034 ?gutenbergId .
  ?book wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= %(min_sitelinks)s)
  OPTIONAL { ?book wdt:P50 ?author }
  OPTIONAL { ?book wdt:P577 ?pubDate }
  OPTIONAL { ?book wdt:P136 ?genre }
  OPTIONAL {
    ?wpArticle schema:about ?book ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?wpTitle .
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY DESC(?sitelinks)
"""

def wikidata_seed(min_sitelinks: int = 15, limit: Optional[int] = 500) -> pd.DataFrame:
    sparql = SPARQLWrapper(WIKIDATA_ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)

    q = WIKIDATA_QUERY % {"min_sitelinks": min_sitelinks}
    if limit is not None:
        q += f"\nLIMIT {int(limit)}"

    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    time.sleep(WD_SLEEP)

    rows = []
    for r in results["results"]["bindings"]:
        pub_raw = r.get("pubDate", {}).get("value", "")
        rows.append({
            "wikidata_entity": r.get("book", {}).get("value", ""),
            "gutenberg_id": int(r["gutenbergId"]["value"]),
            "title": r.get("bookLabel", {}).get("value", "") or "",
            "author": r.get("authorLabel", {}).get("value", "") or "",
            "pub_year": pub_raw[:4] if pub_raw else "",
            "genre_wikidata": r.get("genreLabel", {}).get("value", "") or "",
            "sitelinks": int(r.get("sitelinks", {}).get("value", 0)),
            "wikipedia_title": r.get("wpTitle", {}).get("value", "") or "",
        })

    df = pd.DataFrame(rows).drop_duplicates("gutenberg_id")
    return df


# -------------------------
# 2) Open Library enrichment
# -------------------------

def openlibrary_search_best_match(title: str, author: str = "", pub_year: str = "") -> Optional[dict]:
    q = f'{title} {author}'.strip()
    data = cached_get_json(
        OPENLIB_SEARCH,
        params={"q": q, "limit": 10},
        prefix="ol_search",
        sleep_s=OL_SLEEP
    )
    if not data or "docs" not in data:
        return None
    docs = data["docs"] or []
    if not docs:
        return None

    target_year = int(pub_year) if (pub_year and str(pub_year).isdigit()) else None

    scored: List[Tuple[float, dict]] = []
    for d in docs:
        t = (d.get("title") or "").strip()
        a = " ".join((d.get("author_name") or [])[:2])

        title_score = fuzz.token_set_ratio(title, t) / 100.0 if t else 0.0
        author_score = fuzz.token_set_ratio(author, a) / 100.0 if (author and a) else (0.55 if not author else 0.0)

        year_penalty = 0.0
        if target_year and d.get("first_publish_year"):
            dy = abs(int(d["first_publish_year"]) - target_year)
            year_penalty = min(0.25, dy / 200.0)

        score = 0.65 * title_score + 0.30 * author_score - year_penalty
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_doc = scored[0]
    if best_score < 0.60:
        return None
    return best_doc

def openlibrary_work_details(work_key: str) -> Optional[dict]:
    if not work_key.startswith("/works/"):
        return None
    url = OPENLIB_WORK_JSON.format(work_key=work_key)
    return cached_get_json(url, prefix="ol_work", sleep_s=OL_SLEEP)

def _extract_description(work: dict) -> str:
    desc = work.get("description")
    if isinstance(desc, dict):
        return (desc.get("value") or "").strip()
    if isinstance(desc, str):
        return desc.strip()
    return ""

def openlibrary_enrich_row(row: dict) -> dict:
    title = row.get("title", "")
    author = row.get("author", "")
    pub_year = str(row.get("pub_year", "") or "")

    doc = openlibrary_search_best_match(title, author, pub_year)
    if not doc:
        return {**row,
            "ol_work_key": "",
            "ol_first_publish_year": "",
            "ol_subjects": [],
            "ol_description": "",
            "ol_covers": [],
            "ol_isbn10": [],
            "ol_isbn13": [],
        }

    work_key = (doc.get("key") or "").strip()
    work = openlibrary_work_details(work_key) if work_key else None

    subjects = work.get("subjects") if work else None
    covers = work.get("covers") if work else None

    raw_isbns = [str(x) for x in (doc.get("isbn") or [])]
    isbn10, isbn13 = [], []
    for x in raw_isbns:
        x = re.sub(r"[^0-9Xx]", "", x)
        if len(x) == 10:
            isbn10.append(x.upper())
        elif len(x) == 13:
            isbn13.append(x)

    return {**row,
        "ol_work_key": work_key,
        "ol_first_publish_year": doc.get("first_publish_year", "") or "",
        "ol_subjects": (subjects or [])[:60],
        "ol_description": _extract_description(work or {})[:4000],
        "ol_covers": (covers or [])[:10],
        "ol_isbn10": sorted(set(isbn10)),
        "ol_isbn13": sorted(set(isbn13)),
    }


# -------------------------
# 3) StoryGraph enrichment (public)
# -------------------------

def storygraph_resolve_uuid(title: str, author: str = "", isbn13: Optional[str] = None) -> Optional[str]:
    search_term = isbn13 if isbn13 else f"{title} {author}".strip()
    html = cached_get_text(
        STORYGRAPH_BROWSE,
        params={"search_term": search_term},
        prefix="sg_browse",
        sleep_s=SG_SLEEP
    )
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    links = soup.select('a[href^="/books/"]')
    if not links:
        return None

    candidates: List[Tuple[float, str]] = []
    for a in links:
        href = a.get("href", "")
        m = re.match(r"^/books/([0-9a-fA-F-]{36})", href)
        if not m:
            continue
        uuid = m.group(1)

        link_text = " ".join(a.get_text(" ", strip=True).split())
        if not link_text:
            link_text = a.get("aria-label", "") or ""

        # Mostly title match; author often not present in link text
        tscore = (fuzz.token_set_ratio(title, link_text) / 100.0) if link_text else 0.55
        candidates.append((tscore, uuid))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_uuid = candidates[0]
    if best_score < 0.55:
        return None
    return best_uuid

def storygraph_scrape_book(uuid: str) -> Dict[str, Any]:
    url = STORYGRAPH_BOOK.format(uuid=uuid)
    html = cached_get_text(url, prefix="sg_book", sleep_s=SG_SLEEP)
    if not html:
        return {"sg_uuid": uuid, "sg_url": url, "sg_title": "", "sg_author": "", "sg_tags": []}

    soup = BeautifulSoup(html, "html.parser")

    # Title
    sg_title = ""
    h1 = soup.find("h1")
    if h1:
        sg_title = " ".join(h1.get_text(" ", strip=True).split())

    # Author (heuristic)
    text = soup.get_text("\n", strip=True)
    sg_author = ""
    m = re.search(r"\nby\s+([^\n]+)", "\n" + text, flags=re.IGNORECASE)
    if m:
        sg_author = m.group(1).strip()

    # Tags heuristic: collect short lines with “-paced” or common mood vocabulary
    mood_markers = ("-paced", "dark", "emotional", "lighthearted", "funny", "tense", "reflective", "hopeful", "mysterious")
    tag_lines = []
    for line in text.split("\n"):
        l = line.strip()
        if not l:
            continue
        if any(mm in l.lower() for mm in mood_markers):
            if 5 <= len(l.split()) <= 35:
                tag_lines.append(l)

    tokens: List[str] = []
    for l in tag_lines:
        for tok in re.split(r"\s+", l.strip()):
            tok = tok.strip().lower()
            tok = re.sub(r"[^a-z0-9\-]", "", tok)
            if not tok or len(tok) > 28:
                continue
            tokens.append(tok)

    # cleanup + uniqueness
    bad = {"to", "read", "currently", "reading", "dnf", "finish", "menu", "expand", "dropdown"}
    tokens = [t for t in tokens if t not in bad]
    tokens = sorted(set(tokens))

    return {"sg_uuid": uuid, "sg_url": url, "sg_title": sg_title, "sg_author": sg_author, "sg_tags": tokens[:120]}

def storygraph_enrich_row(row: dict) -> dict:
    isbn13 = None
    if row.get("ol_isbn13"):
        for x in row["ol_isbn13"]:
            x = re.sub(r"[^0-9]", "", str(x))
            if len(x) == 13:
                isbn13 = x
                break

    uuid = storygraph_resolve_uuid(row.get("title", ""), row.get("author", ""), isbn13=isbn13)
    if not uuid:
        return {**row, "sg_uuid": "", "sg_url": "", "sg_title": "", "sg_author": "", "sg_tags": []}

    sg = storygraph_scrape_book(uuid)
    return {**row, **sg}


# -------------------------
# 4) Dedupe layer
# -------------------------

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\u2019']", "", s)             # apostrophes
    s = re.sub(r"[^a-z0-9\s]", " ", s)          # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _work_key(row: dict) -> str:
    """
    Primary key preference:
      - Open Library work key (best)
      - Gutenberg id (good)
      - ISBN13 (ok)
      - normalized title+author (fallback)
    """
    if row.get("ol_work_key"):
        return f"ol:{row['ol_work_key']}"
    if row.get("gutenberg_id"):
        return f"pg:{row['gutenberg_id']}"
    if row.get("ol_isbn13"):
        return f"isbn13:{row['ol_isbn13'][0]}"
    return f"ta:{_norm(row.get('title',''))}::{_norm(row.get('author',''))}"

def dedupe_records(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Returns:
      - merged_works: list of canonical work dicts
      - record_map: list mapping source row -> canonical_work_id
    """
    buckets: Dict[str, List[dict]] = {}
    for r in rows:
        k = _work_key(r)
        buckets.setdefault(k, []).append(r)

    merged = []
    mapping = []
    for k, items in buckets.items():
        # merge strategy: pick the “most complete” item as base, then fill missing
        def completeness(it: dict) -> int:
            score = 0
            for fld in ("ol_work_key", "wikidata_entity", "sg_uuid"):
                score += 2 if it.get(fld) else 0
            score += 1 if it.get("ol_description") else 0
            score += min(3, len(it.get("ol_subjects") or [])) // 1
            score += min(3, len(it.get("sg_tags") or [])) // 1
            score += 1 if it.get("pub_year") else 0
            return score

        items_sorted = sorted(items, key=completeness, reverse=True)
        base = dict(items_sorted[0])

        # union list fields
        def union_list(field: str):
            vals = []
            for it in items_sorted:
                v = it.get(field) or []
                if isinstance(v, list):
                    vals.extend(v)
            # preserve order but unique
            seen = set()
            out = []
            for x in vals:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            base[field] = out

        union_list("ol_subjects")
        union_list("ol_covers")
        union_list("ol_isbn10")
        union_list("ol_isbn13")
        union_list("sg_tags")

        # fill scalar missing values
        for it in items_sorted[1:]:
            for fld in ("title", "author", "pub_year", "genre_wikidata", "wikipedia_title",
                        "wikidata_entity", "ol_work_key", "ol_first_publish_year", "ol_description",
                        "sg_uuid", "sg_url", "sg_title", "sg_author"):
                if not base.get(fld) and it.get(fld):
                    base[fld] = it[fld]

        # canonical id
        canon_id = hashlib.sha1(k.encode("utf-8")).hexdigest()[:16]
        base["work_id"] = canon_id
        base["dedupe_key"] = k
        merged.append(base)

        for it in items:
            mapping.append({
                "work_id": canon_id,
                "source_dedupe_key": k,
                "source_gutenberg_id": it.get("gutenberg_id", None),
                "source_ol_work_key": it.get("ol_work_key", ""),
                "source_wikidata_entity": it.get("wikidata_entity", ""),
                "source_sg_uuid": it.get("sg_uuid", ""),
            })

    return merged, mapping


# -------------------------
# 5) Normalize to SQLite
# -------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS works (
  work_id TEXT PRIMARY KEY,
  title TEXT,
  pub_year TEXT,
  description TEXT,
  genre_wikidata TEXT,
  wikipedia_title TEXT,
  sitelinks INTEGER,
  gutenberg_id INTEGER,
  wikidata_entity TEXT,
  ol_work_key TEXT,
  sg_uuid TEXT,
  sg_url TEXT,
  dedupe_key TEXT
);

CREATE TABLE IF NOT EXISTS authors (
  author_id TEXT PRIMARY KEY,
  name TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS work_authors (
  work_id TEXT,
  author_id TEXT,
  role TEXT DEFAULT 'author',
  PRIMARY KEY (work_id, author_id),
  FOREIGN KEY (work_id) REFERENCES works(work_id),
  FOREIGN KEY (author_id) REFERENCES authors(author_id)
);

CREATE TABLE IF NOT EXISTS subjects (
  subject_id TEXT PRIMARY KEY,
  subject TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS work_subjects (
  work_id TEXT,
  subject_id TEXT,
  PRIMARY KEY (work_id, subject_id),
  FOREIGN KEY (work_id) REFERENCES works(work_id),
  FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

CREATE TABLE IF NOT EXISTS identifiers (
  work_id TEXT,
  id_type TEXT,
  id_value TEXT,
  PRIMARY KEY (work_id, id_type, id_value),
  FOREIGN KEY (work_id) REFERENCES works(work_id)
);

CREATE TABLE IF NOT EXISTS tags (
  tag_id TEXT PRIMARY KEY,
  tag TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS work_tags (
  work_id TEXT,
  tag_id TEXT,
  source TEXT DEFAULT 'storygraph',
  PRIMARY KEY (work_id, tag_id, source),
  FOREIGN KEY (work_id) REFERENCES works(work_id),
  FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
);

CREATE TABLE IF NOT EXISTS record_map (
  work_id TEXT,
  source_dedupe_key TEXT,
  source_gutenberg_id INTEGER,
  source_ol_work_key TEXT,
  source_wikidata_entity TEXT,
  source_sg_uuid TEXT
);
"""

def _id_for(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def write_sqlite(db_path: str, works: List[dict], record_map: List[dict]) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(SCHEMA)

    # Upsert helpers (simple insert ignore)
    def ins(table: str, cols: List[str], rows: List[Tuple]):
        if not rows:
            return
        q = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        cur.executemany(q, rows)

    # Works + Authors + relationships
    work_rows = []
    author_rows = []
    work_author_rows = []

    subject_rows = []
    work_subject_rows = []

    ident_rows = []
    tag_rows = []
    work_tag_rows = []

    for w in works:
        work_rows.append((
            w["work_id"],
            w.get("title",""),
            str(w.get("pub_year","") or ""),
            w.get("ol_description","") or w.get("description","") or "",
            w.get("genre_wikidata","") or "",
            w.get("wikipedia_title","") or "",
            int(w.get("sitelinks") or 0),
            int(w.get("gutenberg_id") or 0) if w.get("gutenberg_id") is not None else None,
            w.get("wikidata_entity","") or "",
            w.get("ol_work_key","") or "",
            w.get("sg_uuid","") or "",
            w.get("sg_url","") or "",
            w.get("dedupe_key","") or "",
        ))

        # Author
        author_name = (w.get("author") or "").strip()
        if author_name:
            aid = _id_for("author:" + author_name.lower())
            author_rows.append((aid, author_name))
            work_author_rows.append((w["work_id"], aid, "author"))

        # Subjects
        for s in (w.get("ol_subjects") or []):
            s = str(s).strip()
            if not s:
                continue
            sid = _id_for("subj:" + s.lower())
            subject_rows.append((sid, s))
            work_subject_rows.append((w["work_id"], sid))

        # Identifiers
        if w.get("gutenberg_id"):
            ident_rows.append((w["work_id"], "gutenberg", str(w["gutenberg_id"])))
        if w.get("ol_work_key"):
            ident_rows.append((w["work_id"], "openlibrary_work", w["ol_work_key"]))
        if w.get("wikidata_entity"):
            ident_rows.append((w["work_id"], "wikidata", w["wikidata_entity"]))
        for x in (w.get("ol_isbn10") or []):
            ident_rows.append((w["work_id"], "isbn10", str(x)))
        for x in (w.get("ol_isbn13") or []):
            ident_rows.append((w["work_id"], "isbn13", str(x)))
        if w.get("sg_uuid"):
            ident_rows.append((w["work_id"], "storygraph_uuid", w["sg_uuid"]))

        # Tags (StoryGraph)
        for t in (w.get("sg_tags") or []):
            t = str(t).strip().lower()
            if not t:
                continue
            tid = _id_for("tag:" + t)
            tag_rows.append((tid, t))
            work_tag_rows.append((w["work_id"], tid, "storygraph"))

    ins("works",
        ["work_id","title","pub_year","description","genre_wikidata","wikipedia_title","sitelinks","gutenberg_id",
         "wikidata_entity","ol_work_key","sg_uuid","sg_url","dedupe_key"],
        work_rows
    )
    ins("authors", ["author_id","name"], author_rows)
    ins("work_authors", ["work_id","author_id","role"], work_author_rows)

    ins("subjects", ["subject_id","subject"], subject_rows)
    ins("work_subjects", ["work_id","subject_id"], work_subject_rows)

    ins("identifiers", ["work_id","id_type","id_value"], ident_rows)

    ins("tags", ["tag_id","tag"], tag_rows)
    ins("work_tags", ["work_id","tag_id","source"], work_tag_rows)

    # record map (not unique)
    cur.executemany(
        "INSERT INTO record_map (work_id,source_dedupe_key,source_gutenberg_id,source_ol_work_key,source_wikidata_entity,source_sg_uuid) VALUES (?,?,?,?,?,?)",
        [(m.get("work_id",""), m.get("source_dedupe_key",""), m.get("source_gutenberg_id",None), m.get("source_ol_work_key",""),
          m.get("source_wikidata_entity",""), m.get("source_sg_uuid","")) for m in record_map]
    )

    con.commit()
    con.close()


def flatten_for_csv(works: List[dict]) -> pd.DataFrame:
    rows = []
    for w in works:
        rows.append({
            "work_id": w["work_id"],
            "title": w.get("title",""),
            "author": w.get("author",""),
            "pub_year": w.get("pub_year",""),
            "description": (w.get("ol_description") or "")[:1500],
            "subjects_json": json.dumps(w.get("ol_subjects") or [], ensure_ascii=False),
            "tags_json": json.dumps(w.get("sg_tags") or [], ensure_ascii=False),
            "gutenberg_id": w.get("gutenberg_id", ""),
            "ol_work_key": w.get("ol_work_key",""),
            "wikidata_entity": w.get("wikidata_entity",""),
            "sg_uuid": w.get("sg_uuid",""),
            "sg_url": w.get("sg_url",""),
        })
    return pd.DataFrame(rows)


# -------------------------
# Pipeline
# -------------------------

def build_catalog(min_sitelinks: int, limit: int, do_storygraph: bool) -> Tuple[List[dict], List[dict]]:
    print("1) Wikidata seed…")
    seed = wikidata_seed(min_sitelinks=min_sitelinks, limit=limit)
    print(f"Seeded {len(seed)} rows")

    print("2) Open Library enrichment…")
    enriched: List[dict] = []
    for _, r in tqdm(seed.iterrows(), total=len(seed)):
        enriched.append(openlibrary_enrich_row(r.to_dict()))

    if do_storygraph:
        print("3) StoryGraph enrichment (optional)…")
        out: List[dict] = []
        for r in tqdm(enriched, total=len(enriched)):
            out.append(storygraph_enrich_row(r))
        enriched = out

    print("4) Dedupe…")
    merged, mapping = dedupe_records(enriched)
    print(f"Canonical works: {len(merged)} (from {len(enriched)} input rows)")
    return merged, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "catalog.db"), help="Output SQLite path")
    ap.add_argument("--csv", default=os.path.join(DATA_DIR, "catalog_flat.csv"), help="Output flattened CSV path")
    ap.add_argument("--limit", type=int, default=300, help="Wikidata LIMIT")
    ap.add_argument("--min-sitelinks", type=int, default=15, help="Wikidata sitelinks threshold")
    ap.add_argument("--storygraph", type=int, default=1, help="1 to enable StoryGraph enrichment, 0 to disable")
    args = ap.parse_args()

    works, record_map = build_catalog(
        min_sitelinks=args.min_sitelinks,
        limit=args.limit,
        do_storygraph=bool(args.storygraph)
    )

    print("5) Write SQLite…")
    write_sqlite(args.out, works, record_map)

    print("6) Write flattened CSV…")
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    df_flat = flatten_for_csv(works)
    df_flat.to_csv(args.csv, index=False)

    print(f"Done.\nSQLite: {args.out}\nCSV:    {args.csv}\nCache:  {CACHE_DIR}")


if __name__ == "__main__":
    main()
