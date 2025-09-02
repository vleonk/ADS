from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter

import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 1) Load Data
# ---------------------------

DATA_DIR = Path(r"..") / "ADS" / "Datasets"
df_news = pd.read_csv(DATA_DIR / "NLP_data.csv")

# 2) Quick EDA
# ---------------------------

print("\n=== Shapes ===\n")
print(df_news.shape)

print("\n=== Info ===\n")
print(df_news.info())

print("\n=== Describe ===\n")
print(df_news.describe(include="object").T)

print("\n=== Missing Data ===\n")
mis = df_news.isna().sum()
mis = mis[mis > 0].sort_values(ascending=False)
missing_tbl = (pd.DataFrame({"column": mis.index, "n_missing": mis.values})
               .assign(pct=lambda d: (d["n_missing"] / len(df_news) * 100).round(2))
               .reset_index(drop=True))
print(missing_tbl)

print("\n=== Duplicates (by article_id / url) ===\n")
dup_ids = df_news["article_id"].duplicated().sum() if "article_id" in df_news.columns else 0
dup_urls = df_news["url"].duplicated().sum() if "url" in df_news.columns else 0
print({"article_id_dupes": int(dup_ids), "url_dupes": int(dup_urls)})

# 3) Cleaning Data
# ---------------------------

# Parse dates
df_news["published_at"] = pd.to_datetime(df_news["published_at"], errors="coerce", utc=True)

# Normalize a few string cols commonly inspected
for col in ["category", "source_name", "author"]:
    df_news[col] = df_news[col].astype(str).str.strip()

# Drop duplicate
df_news = df_news.sort_values("published_at", na_position="last").drop_duplicates("article_id", keep="last")
df_news = df_news.drop_duplicates("url", keep="first")

# Build a canonical text field (title + description + full_content/content)
text_cols = [c for c in ["title", "description", "full_content", "content"]]
df_news["raw_text"] = df_news[text_cols].fillna("").agg(" ".join, axis=1).str.strip()

# Lowercasing text
df_news["raw_text"] = df_news["raw_text"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()

# Removing URLs/HTML/Non-alpha using regex
df_news["raw_text"] = (
    df_news["raw_text"].astype(str)
        .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)  # Links
        .str.replace(r"<[^>]+>", " ", regex=True)                # HTML tags
        .str.replace(r"[^A-Za-z\s]", " ", regex=True)            # Non-letters
        .str.replace(r"\s+", " ", regex=True)                    # Extra spaces
        .str.strip()
)

# 4) Text Normalization
# ---------------------------

nltk.download('stopwords')
stop_words = stopwords.words('english')

# Stem each token
stemmer = PorterStemmer()

def process_text(text: str):
    if not isinstance(text, str):
        text = ""
    tokens = word_tokenize(text, preserve_line=True)
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    return [stemmer.stem(w) for w in tokens]

# Apply to every row
df_news["stems"] = df_news["raw_text"].fillna("").apply(process_text)

# Saving cleaned df
# df_news.to_csv("clean_nlp.csv", index=False)

# 5) Visualize top 25 stem words
# ---------------------------

if "stems" in df_news.columns:
    all_stems = (stem for lst in df_news["stems"] if isinstance(lst, list) for stem in lst)
    stem_counts = Counter(all_stems)
    if stem_counts:
        top_n = 25
        most = stem_counts.most_common(top_n)
        stems, counts = zip(*most)
        fig, ax = plt.subplots(figsize=(7,6))
        ax.barh(stems[::-1], counts[::-1], color="#40AFFF")
        ax.set_title(f"Top {top_n} stems")
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        plt.show()

# 6) Visualize articles per day
# ---------------------------

# Ensure we have a datetime column called published_dt
if "published_dt" not in df_news.columns:
    DATE_CANDIDATES = ["published_at", "published", "date", "pub_date", "datetime", "timestamp", "created_at"]
    DATE_COL = next((c for c in DATE_CANDIDATES if c in df_news.columns), None)
    if DATE_COL is None:
        raise ValueError("Set DATE_COL to your datetime column (e.g., DATE_COL = 'published_date').")
    df_news["published_dt"] = pd.to_datetime(df_news[DATE_COL], errors="coerce")

# Drop rows where parsing failed
df_news = df_news.dropna(subset=["published_dt"])

# Daily counts (fill missing days with 0)
daily_counts = (
    df_news.set_index("published_dt")
           .resample("D")
           .size()
           .asfreq("D", fill_value=0)
)

start = daily_counts.index.min().strftime("%Y-%m-%d")
end   = daily_counts.index.max().strftime("%Y-%m-%d")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily_counts.index, daily_counts.values, marker="o", linewidth=1)
ax.set_title(f"Articles per Day ({start} to {end})")
ax.set_xlabel("")
ax.set_ylabel("Number of articles")

# Use a regular DateFormatter to avoid the month/year summary label
locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))  # e.g., "Oct 05"
plt.xticks(rotation=45)
plt.margins(x=0)
plt.tight_layout()
plt.show()

