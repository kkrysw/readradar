"""
scraper.py
==========
Scrapes book metadata, ratings, and reviews from Goodreads,
and supplementary metadata from Open Library.

Usage:
    python scraper.py --mode isbn       --input isbns.txt
    python scraper.py --mode search     --query "brandon sanderson stormlight"
    python scraper.py --mode author     --query "Ursula K. Le Guin"
    python scraper.py --mode openlibrary --query "dune frank herbert"
"""

import re
import time
import json
import random
import logging
import argparse
import sqlite3
import requests

from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOODREADS_BASE   = "https://www.goodreads.com"
OPENLIBRARY_BASE = "https://openlibrary.org"
GUTENBERG_BASE   = "https://gutendex.com"        # REST wrapper for Gutenberg

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# How many reviews to collect per book
MAX_REVIEWS        = 100
# Seconds to sleep between requests (randomized around this value)
BASE_SLEEP         = 2.5
# Max retries on transient failures
MAX_RETRIES        = 3

# ---------------------------------------------------------------------------
# Data classes  (these become the DB schema rows)
# ---------------------------------------------------------------------------

@dataclass
class Book:
    book_id:           str                    # Goodreads book ID
    title:             str
    author:            str
    isbn:              Optional[str]          = None
    description:       Optional[str]          = None
    page_count:        Optional[int]          = None
    publication_year:  Optional[int]          = None
    average_rating:    Optional[float]        = None
    # Stored as JSON string  {"1":n,"2":n,"3":n,"4":n,"5":n}
    rating_distribution: Optional[str]        = None
    genres:            Optional[str]          = None   # JSON list
    series_name:       Optional[str]          = None
    series_number:     Optional[float]        = None
    cover_url:         Optional[str]          = None
    goodreads_url:     Optional[str]          = None
    open_library_id:   Optional[str]          = None
    gutenberg_id:      Optional[int]          = None
    scraped_at:        str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class Review:
    review_id:         str                    # Goodreads review ID
    book_id:           str                    # FK → Book.book_id
    reviewer_username: str
    star_rating:       Optional[int]          = None   # 1-5; None if no rating given
    review_text:       Optional[str]          = None
    date_posted:       Optional[str]          = None
    likes_count:       int                    = 0
    is_spoiler:        bool                   = False
    scraped_at:        str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(path: str = "books.db") -> sqlite3.Connection:
    """Create tables if they don't exist and return the connection."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS books (
            book_id             TEXT PRIMARY KEY,
            title               TEXT,
            author              TEXT,
            isbn                TEXT,
            description         TEXT,
            page_count          INTEGER,
            publication_year    INTEGER,
            average_rating      REAL,
            rating_distribution TEXT,
            genres              TEXT,
            series_name         TEXT,
            series_number       REAL,
            cover_url           TEXT,
            goodreads_url       TEXT,
            open_library_id     TEXT,
            gutenberg_id        INTEGER,
            scraped_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS reviews (
            review_id           TEXT PRIMARY KEY,
            book_id             TEXT,
            reviewer_username   TEXT,
            star_rating         INTEGER,
            review_text         TEXT,
            date_posted         TEXT,
            likes_count         INTEGER DEFAULT 0,
            is_spoiler          INTEGER DEFAULT 0,
            scraped_at          TEXT,
            FOREIGN KEY (book_id) REFERENCES books(book_id)
        );
    """)
    conn.commit()
    log.info("Database initialised at %s", path)
    return conn


def upsert_book(conn: sqlite3.Connection, book: Book):
    d = asdict(book)
    placeholders = ", ".join(["?"] * len(d))
    columns      = ", ".join(d.keys())
    updates      = ", ".join(f"{k}=excluded.{k}" for k in d if k != "book_id")
    conn.execute(
        f"INSERT INTO books ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(book_id) DO UPDATE SET {updates}",
        list(d.values())
    )
    conn.commit()


def upsert_review(conn: sqlite3.Connection, review: Review):
    d = asdict(review)
    d["is_spoiler"] = int(d["is_spoiler"])
    placeholders = ", ".join(["?"] * len(d))
    columns      = ", ".join(d.keys())
    updates      = ", ".join(f"{k}=excluded.{k}" for k in d if k != "review_id")
    conn.execute(
        f"INSERT INTO reviews ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(review_id) DO UPDATE SET {updates}",
        list(d.values())
    )
    conn.commit()


def already_scraped(conn: sqlite3.Connection, book_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM books WHERE book_id=? AND title IS NOT NULL", (book_id,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Rate-limiting helpers
# ---------------------------------------------------------------------------

def polite_sleep(base: float = BASE_SLEEP):
    """Sleep for base ± 40% to avoid appearing robotic."""
    jitter = base * 0.4 * (random.random() * 2 - 1)
    time.sleep(max(0.5, base + jitter))


def retry(fn, *args, retries=MAX_RETRIES, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                polite_sleep(BASE_SLEEP * attempt)   # back-off
    log.error("All %d attempts failed for %s", retries, fn.__name__)
    return None


# ---------------------------------------------------------------------------
# Playwright browser context (shared across calls)
# ---------------------------------------------------------------------------

class BrowserContext:
    """
    Thin wrapper so we can reuse a single Playwright browser
    instance across multiple scrape calls.
    """
    def __init__(self):
        self._playwright = None
        self._browser    = None
        self._context    = None

    def start(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="en-US",
            viewport={"width": 1280, "height": 900},
        )
        log.info("Browser started.")

    def new_page(self):
        return self._context.new_page()

    def stop(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        log.info("Browser stopped.")


# ---------------------------------------------------------------------------
# Goodreads search  →  list of (book_id, url)
# ---------------------------------------------------------------------------

def search_goodreads(query: str, browser: BrowserContext, max_results: int = 5):
    """
    Returns up to max_results (book_id, full_url) pairs from the search page.
    Uses requests + BS4 since the search results page is server-rendered.
    """
    url = f"{GOODREADS_BASE}/search?q={requests.utils.quote(query)}"
    log.info("Searching Goodreads: %s", url)

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for a in soup.select("a.bookTitle")[:max_results]:
        href = a.get("href", "")
        # href looks like  /book/show/12345-book-title
        m = re.search(r"/book/show/(\d+)", href)
        if m:
            book_id = m.group(1)
            results.append((book_id, GOODREADS_BASE + href))

    log.info("Found %d results.", len(results))
    return results


# ---------------------------------------------------------------------------
# Goodreads book page  →  Book dataclass
# ---------------------------------------------------------------------------

def _extract_book_id_from_url(url: str) -> str:
    m = re.search(r"/book/show/(\d+)", url)
    return m.group(1) if m else url


def scrape_book_page(url: str, browser: BrowserContext) -> Optional[Book]:
    """
    Scrapes the main book page. Uses Playwright because key metadata
    (rating distribution, genres) lives in the __NEXT_DATA__ JSON blob
    that requires JS execution to fully hydrate.
    """
    book_id = _extract_book_id_from_url(url)
    log.info("Scraping book page: %s (id=%s)", url, book_id)

    page = browser.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector("h1[data-testid='bookTitle'], h1.Text__title1", timeout=15_000)
        html = page.content()
    except PlaywrightTimeout:
        log.warning("Timeout loading book page %s", url)
        return None
    finally:
        page.close()

    soup = BeautifulSoup(html, "html.parser")

    # ── Pull structured data from __NEXT_DATA__ JSON blob ──────────────────
    next_data = _parse_next_data(soup)
    apollo     = next_data.get("props", {}).get("pageProps", {}).get("apolloState", {}) if next_data else {}

    # ── Title ───────────────────────────────────────────────────────────────
    title_el = (
        soup.select_one("h1[data-testid='bookTitle']") or
        soup.select_one("h1.Text__title1")
    )
    title = title_el.get_text(strip=True) if title_el else "Unknown"

    # ── Author ──────────────────────────────────────────────────────────────
    author_el = soup.select_one("span.ContributorLink__name")
    author = author_el.get_text(strip=True) if author_el else "Unknown"

    # ── Description ─────────────────────────────────────────────────────────
    desc_el = soup.select_one("span.Formatted")
    description = None
    if desc_el:
        description = (
            desc_el.get_text(separator="\n")
                   .replace("\u00a0", " ")
                   .strip()
        )

    # ── Page count & publication year ───────────────────────────────────────
    page_count, pub_year = _extract_featured_details(soup)

    # ── Average rating ──────────────────────────────────────────────────────
    avg_el = soup.select_one("div.RatingStatistics__rating")
    average_rating = float(avg_el.get_text(strip=True)) if avg_el else None

    # ── Rating distribution (1-5 star breakdown) ────────────────────────────
    rating_distribution = _extract_rating_distribution(soup, apollo)

    # ── Genres (first 4) ────────────────────────────────────────────────────
    genres = []
    for el in soup.select(".BookPageMetadataSection__genreButton .Button__labelItem")[:4]:
        genres.append(el.get_text(strip=True))

    # ── Series ──────────────────────────────────────────────────────────────
    series_name, series_number = _extract_series(soup)

    # ── Cover image ─────────────────────────────────────────────────────────
    cover_el = soup.select_one(".BookCover__image img.ResponsiveImage")
    cover_url = cover_el["src"] if cover_el else None

    # ── ISBN ─────────────────────────────────────────────────────────────────
    isbn = _extract_isbn(soup, apollo)

    return Book(
        book_id            = book_id,
        title              = title,
        author             = author,
        isbn               = isbn,
        description        = description,
        page_count         = page_count,
        publication_year   = pub_year,
        average_rating     = average_rating,
        rating_distribution= json.dumps(rating_distribution) if rating_distribution else None,
        genres             = json.dumps(genres),
        series_name        = series_name,
        series_number      = series_number,
        cover_url          = cover_url,
        goodreads_url      = url,
    )


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_next_data(soup: BeautifulSoup) -> Optional[dict]:
    el = soup.select_one("#__NEXT_DATA__")
    if not el:
        return None
    try:
        return json.loads(el.string)
    except json.JSONDecodeError:
        return None


def _extract_featured_details(soup: BeautifulSoup):
    page_count  = None
    pub_year    = None
    details     = soup.select_one(".FeaturedDetails")
    if details:
        pages_el = details.select_one('p[data-testid="pagesFormat"]')
        if pages_el:
            m = re.search(r"[\d,]+", pages_el.get_text())
            if m:
                page_count = int(m.group().replace(",", ""))

        pub_el = details.select_one('p[data-testid="publicationInfo"]')
        if pub_el:
            m = re.search(r"\b(\d{4})\b", pub_el.get_text())
            if m:
                pub_year = int(m.group(1))
    return page_count, pub_year


def _extract_rating_distribution(soup: BeautifulSoup, apollo: dict) -> Optional[dict]:
    """
    Try to get the 1-5 star counts from the Apollo cache first,
    then fall back to the rendered histogram bars.
    """
    # Apollo path
    for val in apollo.values():
        if isinstance(val, dict) and "ratingDistribution" in val:
            raw = val["ratingDistribution"]
            # raw is usually a list of {starRating, count}
            if isinstance(raw, list):
                return {str(item["starRating"]): item["count"] for item in raw}

    # Fallback: histogram bars have aria-label="X star: Y ratings"
    dist = {}
    for bar in soup.select("[aria-label*='star']"):
        label = bar.get("aria-label", "")
        m = re.match(r"(\d) star[s]?: ([\d,]+)", label)
        if m:
            dist[m.group(1)] = int(m.group(2).replace(",", ""))
    return dist if dist else None


def _extract_series(soup: BeautifulSoup):
    series_el = soup.select_one(".BookPageTitleSection__title a")
    if not series_el:
        return None, None
    text = series_el.get_text(strip=True)
    m = re.match(r"^(.*?)\s*#?([\d.]+)$", text)
    if m:
        return m.group(1).strip(), float(m.group(2))
    return text, None


def _extract_isbn(soup: BeautifulSoup, apollo: dict) -> Optional[str]:
    # Try Apollo state
    for val in apollo.values():
        if isinstance(val, dict):
            isbn = val.get("isbn13") or val.get("isbn")
            if isbn:
                return str(isbn)
    # Fallback: sometimes visible in the edition details
    for dt in soup.select("dt"):
        if "ISBN" in dt.get_text():
            dd = dt.find_next_sibling("dd")
            if dd:
                return dd.get_text(strip=True)
    return None


# ---------------------------------------------------------------------------
# Goodreads reviews  →  list[Review]
# ---------------------------------------------------------------------------

def scrape_reviews(book_id: str, book_url: str,
                   browser: BrowserContext,
                   max_reviews: int = MAX_REVIEWS) -> list[Review]:
    """
    Scrolls through the community reviews section using Playwright
    (JS-rendered) and collects up to max_reviews Review objects.
    """
    log.info("Scraping reviews for book_id=%s", book_id)
    reviews_url = f"{GOODREADS_BASE}/book/show/{book_id}#reviews"

    page = browser.new_page()
    collected: list[Review] = []

    try:
        page.goto(reviews_url, wait_until="domcontentloaded", timeout=30_000)
        # Wait for at least one review card to appear
        page.wait_for_selector(".ReviewCard", timeout=15_000)

        seen_ids: set[str] = set()
        scroll_attempts = 0
        max_scrolls     = 20    # safety cap

        while len(collected) < max_reviews and scroll_attempts < max_scrolls:
            html  = page.content()
            soup  = BeautifulSoup(html, "html.parser")
            cards = soup.select(".ReviewCard")

            for card in cards:
                review = _parse_review_card(card, book_id)
                if review and review.review_id not in seen_ids:
                    seen_ids.add(review.review_id)
                    collected.append(review)
                    if len(collected) >= max_reviews:
                        break

            # Try to load more reviews
            more_btn = page.query_selector("button[aria-label='Show more reviews']")
            if more_btn:
                more_btn.click()
                polite_sleep(2.0)
            else:
                # Scroll down to trigger lazy-load
                page.evaluate("window.scrollBy(0, 1200)")
                polite_sleep(2.5)
                scroll_attempts += 1

    except PlaywrightTimeout:
        log.warning("Timeout while collecting reviews for %s", book_id)
    finally:
        page.close()

    log.info("Collected %d reviews for book_id=%s", len(collected), book_id)
    return collected


def _parse_review_card(card, book_id: str) -> Optional[Review]:
    """Parse a single .ReviewCard element into a Review dataclass."""

    # ── Review ID ──────────────────────────────────────────────────────────
    review_id = card.get("data-review-id") or card.get("id") or ""
    if not review_id:
        # Fall back to constructing from reviewer + date
        review_id = f"{book_id}_{hash(card.get_text()[:80])}"

    # ── Reviewer username ──────────────────────────────────────────────────
    user_el = card.select_one(".ReviewerProfile__name a, .ReviewCard__reviewer a")
    reviewer_username = user_el.get_text(strip=True) if user_el else "anonymous"

    # ── Star rating ────────────────────────────────────────────────────────
    star_rating = None
    rating_el = card.select_one(".RatingStars, [aria-label*='rating']")
    if rating_el:
        label = rating_el.get("aria-label", "")
        m = re.search(r"(\d) out of 5", label)
        if m:
            star_rating = int(m.group(1))

    # ── Review text ────────────────────────────────────────────────────────
    spoiler = False
    text_el = card.select_one(".ReviewText__content, .Formatted")
    review_text = None
    if text_el:
        # Check for spoiler wrapper
        if card.select_one(".SpoilerAlert, [data-spoiler]"):
            spoiler = True
        review_text = (
            text_el.get_text(separator="\n")
                   .replace("\u00a0", " ")
                   .strip()
        )

    # ── Date ───────────────────────────────────────────────────────────────
    date_el = card.select_one("time, .ReviewCard__date")
    date_posted = None
    if date_el:
        date_posted = date_el.get("datetime") or date_el.get_text(strip=True)

    # ── Likes ──────────────────────────────────────────────────────────────
    likes_count = 0
    likes_el = card.select_one(".ReviewCard__like-count, [data-testid='like-count']")
    if likes_el:
        m = re.search(r"\d+", likes_el.get_text())
        if m:
            likes_count = int(m.group())

    return Review(
        review_id         = str(review_id),
        book_id           = book_id,
        reviewer_username = reviewer_username,
        star_rating       = star_rating,
        review_text       = review_text,
        date_posted       = date_posted,
        likes_count       = likes_count,
        is_spoiler        = spoiler,
    )


# ---------------------------------------------------------------------------
# Open Library enrichment  (no scraping — clean JSON API)
# ---------------------------------------------------------------------------

def enrich_from_open_library(book: Book) -> Book:
    """
    Fetches supplementary data from Open Library:
    - Subjects (richer than Goodreads genres)
    - Open Library ID
    - Gutenberg ID (if available)
    Uses the search API since we may not have a clean OL key.
    """
    query = f"{book.title} {book.author}"
    url   = f"{OPENLIBRARY_BASE}/search.json?q={requests.utils.quote(query)}&limit=1&fields=key,subject,id_project_gutenberg,editions"

    log.info("Open Library lookup: %s", query)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Open Library request failed: %s", exc)
        return book

    docs = data.get("docs", [])
    if not docs:
        return book

    doc = docs[0]
    book.open_library_id = doc.get("key")

    # Gutenberg IDs come back as a list; take the first
    gut_ids = doc.get("id_project_gutenberg", [])
    if gut_ids:
        try:
            book.gutenberg_id = int(gut_ids[0])
        except ValueError:
            pass

    # Merge OL subjects with existing genres (deduplicated)
    subjects = doc.get("subject", [])[:10]     # cap at 10
    existing = json.loads(book.genres or "[]")
    merged   = list(dict.fromkeys(existing + subjects))  # preserve order, dedup
    book.genres = json.dumps(merged)

    polite_sleep(1.0)
    return book


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def scrape_books(queries: list[str], mode: str,
                 db_path: str = "books.db",
                 max_books: int = 10,
                 max_reviews: int = MAX_REVIEWS):
    """
    Main entry point.

    mode: 'search' | 'isbn' | 'author'
    """
    conn    = init_db(db_path)
    browser = BrowserContext()
    browser.start()

    try:
        for query in queries:
            log.info("=== Processing query: %s ===", query)

            if mode == "isbn":
                # Treat the query directly as an ISBN search on GR
                results = search_goodreads(query, browser, max_results=1)
            else:
                results = search_goodreads(query, browser, max_results=max_books)

            for book_id, book_url in results:
                if already_scraped(conn, book_id):
                    log.info("Book %s already in DB, skipping.", book_id)
                    continue

                # 1. Scrape main book page
                book = retry(scrape_book_page, book_url, browser)
                if not book:
                    continue

                # 2. Enrich from Open Library
                book = enrich_from_open_library(book)

                # 3. Persist book
                upsert_book(conn, book)
                log.info("Saved book: %s [%s]", book.title, book.book_id)

                # 4. Scrape reviews
                reviews = retry(scrape_reviews, book_id, book_url, browser, max_reviews)
                if reviews:
                    for review in reviews:
                        upsert_review(conn, review)
                    log.info("Saved %d reviews for %s.", len(reviews), book.title)

                polite_sleep(BASE_SLEEP)

    finally:
        browser.stop()
        conn.close()
        log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(description="Goodreads + Open Library scraper")
    p.add_argument("--mode",  choices=["search","isbn","author","openlibrary"],
                   required=True)
    p.add_argument("--query", help="Single search query")
    p.add_argument("--input", help="Path to a text file with one query per line")
    p.add_argument("--db",    default="books.db", help="SQLite database path")
    p.add_argument("--max-books",   type=int, default=10)
    p.add_argument("--max-reviews", type=int, default=MAX_REVIEWS)
    return p


def main():
    args = _build_parser().parse_args()

    if args.input:
        with open(args.input) as f:
            queries = [line.strip() for line in f if line.strip()]
    elif args.query:
        queries = [args.query]
    else:
        raise ValueError("Provide --query or --input")

    scrape_books(
        queries     = queries,
        mode        = args.mode,
        db_path     = args.db,
        max_books   = args.max_books,
        max_reviews = args.max_reviews,
    )


if __name__ == "__main__":
    main()
