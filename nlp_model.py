import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# 1) Load Data
# ---------------------------

df_news = pd.read_csv("clean_nlp.csv")

# 2) Feature Engineering
# ---------------------------

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9,
)

X_tfidf = tfidf.fit_transform(df_news["stems"])
vocab = np.array(tfidf.get_feature_names_out())

print(f"\nTF-IDF shape: {X_tfidf.shape}")

# 3) Topic Modeling
# ---------------------------

nmf = NMF(n_components=10, random_state=42, init="nndsvd", max_iter=400)
W = nmf.fit_transform(X_tfidf)   # doc-topic weights
H = nmf.components_              # topic-term weights

def top_terms_for_topic(H_row, n=10):
    idx = np.argsort(H_row)[::-1][:n]
    return vocab[idx]

topic_terms = []
for k in range(10):
    words = top_terms_for_topic(H[k], n=8)
    topic_terms.append({"topic_id": k, "top_terms": ", ".join(words)})

topics_df = pd.DataFrame(topic_terms)

print("\n=== NMF topics ===\n")
print(topics_df[['topic_id', 'top_terms']].to_string(index=False))

# 4) Sentiment
# ---------------------------

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

def vader_label(compound, pos_thresh=0.05, neg_thresh=-0.05):
    if compound >= pos_thresh:
        return "positive"
    elif compound <= neg_thresh:
        return "negative"
    return "neutral"

vader_scores = df_news["raw_text"].astype(str).apply(sia.polarity_scores)
df_news["sent_compound"] = vader_scores.apply(lambda d: d["compound"])
df_news["sent_label"] = df_news["sent_compound"].apply(vader_label)

print("\n=== Article Sentiment ===")
display(df_news[["article_id","sent_label"]].T)

# 5) Visualize nmf.components_ and top words of Topic 0 
# ---------------------------

words = [vocab[0], vocab[1], vocab[-1]]
idxs  = [0, 1, len(vocab)-1]

w = H[0].astype(float)  # nmf.components_ for topic 0

# One-row table showing first 2 and last words of the topic 0
row = {"topic_id": 0, words[0]: w[idxs[0]], words[1]: w[idxs[1]], words[2]: w[idxs[2]]}
table = pd.DataFrame([row])
table.loc[:, words] = table.loc[:, words].round(6)

print("\n=== nmf.components_ in Topic 0 ===\n")
print(table)

# Top 8 words for topic 0
top_idx = np.argsort(w)[::-1][:8]
top8 = pd.DataFrame({
    "rank": np.arange(1, 9),
    "word": vocab[top_idx],
    "weight": np.round(w[top_idx], 6)
})
print("\n=== Top 8 Words in Topic 0 ===\n")
print(top8)  

# Spike plot for topic 0
plt.figure(figsize=(10, 3))
plt.plot(w, linewidth=0.8)
plt.scatter(top_idx, w[top_idx], s=18)   # mark top-8
plt.title("Topic 0 - nmf.components_ across vocabulary")
plt.xlabel("vocabulary index")
plt.ylabel("weight")
plt.margins(x=0)
plt.tight_layout()
plt.show()

# 6) Visualize sentiment distribution
# ---------------------------

order = ["positive", "neutral", "negative"]
counts = df_news["sent_label"].value_counts().reindex(order, fill_value=0)
perc = (counts / counts.sum() * 100).round(1)

max_count = counts.max() if counts.sum() > 0 else 1

plt.figure(figsize=(7, 4))
bars = plt.barh(counts.index, counts.values, color="#40AFFF")

plt.title("Article Sentiment Distribution")
plt.xlabel("Number of articles")
plt.ylabel("Sentiment")
plt.xlim(0, max_count * 1.25)  # room for labels

# Annotate bars with count and %
for y, (cnt, p) in enumerate(zip(counts.values, perc.values)):
    plt.text(cnt, y, f" {cnt} ({p}%)", va="center", ha="left", fontsize=10)
plt.tight_layout()
plt.show()

# 7) Sentiment over time (line graph)
# ------------------------------------
date_col = "published_at"

# Parse and sort by datetime
df_news["dt"] = pd.to_datetime(df_news[date_col], errors="coerce")
df_news = df_news.dropna(subset=["dt"]).sort_values("dt")

# Build a complete daily date range to include zero-article days
date_idx = pd.date_range(df_news["dt"].min().normalize(),
                         df_news["dt"].max().normalize(),
                         freq="D")

# Daily counts of each sentiment label (pos/neu/neg), including zero days
daily_counts = (
    df_news
      .groupby([pd.Grouper(key="dt", freq="D"), "sent_label"])
      .size()
      .unstack(fill_value=0)
      .reindex(date_idx, fill_value=0)  # ensure full date coverage
      .reindex(columns=["positive", "neutral", "negative"], fill_value=0)  # consistent column order
)

# Plot one line per sentiment (raw counts)
plt.figure(figsize=(10, 4))
plt.plot(daily_counts.index, daily_counts["positive"], linewidth=2, label="Positive", color="#40AFFF")
plt.plot(daily_counts.index, daily_counts["neutral"],  linewidth=2, label="Neutral", color="#D9EFFF")
plt.plot(daily_counts.index, daily_counts["negative"], linewidth=2, label="Negative", color="#005D9F")

plt.title("Article Sentiment Over Time")
plt.xlabel("")
plt.ylabel("Articles per day")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()