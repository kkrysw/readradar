import pandas as pd

# Load data
books = pd.read_parquet("data/processed/books.parquet")
contro = pd.read_parquet("data/artifacts/controversy_final.parquet")

# Use controversy file to define the sampled 5000-book set
sample_ids = contro[["book_id"]]

# Select metadata from full books table
meta_cols = [
    "book_id",
    "title",
    "description",
    "average_rating",
    "ratings_count",
    "image_url",
    "publication_year",
    "num_pages",
    "language_code",
    "publisher",
    "text_reviews_count"
]
books_meta = books[meta_cols]

# Filter full metadata down to the sampled 5000 books
sample_meta = sample_ids.merge(
    books_meta,
    on="book_id",
    how="left",
    sort=False
)

# Keep only the controversy fields you want to add
contro_cols = ["book_id", "top_tags", "overall_judgment"]
contro_small = contro[contro_cols].drop_duplicates("book_id")

# Merge
ui_cache = sample_meta.merge(
    contro_small,
    on="book_id",
    how="left",
    sort=False
)

# Save
ui_cache.to_parquet("data/artifacts/ui_books_cache.parquet", index=False)

print("UI cache built:", ui_cache.shape)
print(ui_cache.head())