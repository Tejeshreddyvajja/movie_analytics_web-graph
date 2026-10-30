"""
web_analytics.py
Digital Web Analytics & Audience Engagement Intelligence module for IMDb movie data.
Calculates:
  - Audience Engagement Rate (Review-to-Vote conversion %)
  - Web Virality Index (Composite algorithmic score)
  - Critic vs Web Audience Sentiment Divergence (Metascore vs IMDb Rating 4-Quadrant Matrix)
  - Web Traffic & Voting Funnel stages
  - Decade-by-decade digital behavior shifts
  - High-impact web outlier detection (Audience Champions vs Critic Darlings)
"""

import pandas as pd
import numpy as np


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Safely converts string/object numbers with commas and dirty text into clean numeric float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


def enrich_web_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches dataframe with cleaned numeric fields and web analytics indicators.
    Guarantees 'metadata' exists alongside 'metascore' for full backwards-compatibility.
    """
    df = df.copy()

    # Clean numeric fields
    if "votes" in df.columns:
        df["votes_num"] = clean_numeric_series(df["votes"]).fillna(0).astype(int)
    else:
        df["votes_num"] = 0

    if "review_count" in df.columns:
        df["reviews_num"] = clean_numeric_series(df["review_count"]).fillna(0).astype(int)
    else:
        df["reviews_num"] = 0

    if "rating" in df.columns:
        df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    else:
        df["rating_num"] = np.nan

    # Metascore / metadata handling
    if "metascore" in df.columns:
        df["metascore_num"] = pd.to_numeric(df["metascore"], errors="coerce")
    elif "metadata" in df.columns:
        df["metascore_num"] = pd.to_numeric(df["metadata"], errors="coerce")
    else:
        df["metascore_num"] = np.nan

    # Maintain metadata column for existing code
    df["metadata"] = df["metascore_num"]

    if "duration" in df.columns:
        df["duration_num"] = clean_numeric_series(df["duration"])
    else:
        df["duration_num"] = np.nan

    if "year" in df.columns:
        df["year_num"] = clean_numeric_series(df["year"])
        # Decade grouping
        df["decade"] = (df["year_num"] // 10 * 10).astype("Int64").astype(str) + "s"
    else:
        df["year_num"] = np.nan
        df["decade"] = "Unknown"

    # 1. Audience Engagement Rate (Review to Vote conversion percentage)
    # Only calculate for titles with at least 50 votes to avoid small-sample distortion
    valid_vote_mask = df["votes_num"] >= 50
    df["engagement_rate"] = np.where(
        valid_vote_mask,
        np.clip((df["reviews_num"] / df["votes_num"]) * 100, 0, 100),
        np.nan
    ).round(2)

    # 2. Web Virality Index (Composite normalized 0 - 100 score)
    # 45% Log(Votes) + 35% Log(Reviews) + 20% Rating
    log_votes = np.log10(df["votes_num"].clip(lower=1))
    max_log_v = log_votes.max() if log_votes.max() > 0 else 1.0

    log_reviews = np.log10(df["reviews_num"].clip(lower=1))
    max_log_r = log_reviews.max() if log_reviews.max() > 0 else 1.0

    # Fill rating for virality computation
    fill_rating = df["rating_num"].fillna(df["rating_num"].median() if df["rating_num"].notnull().any() else 6.0)

    virality = (
        (log_votes / max_log_v * 45) +
        (log_reviews / max_log_r * 35) +
        (fill_rating / 10.0 * 20)
    )
    df["virality_index"] = virality.round(1)

    # 3. Sentiment Divergence: Audience Rating (scaled to 100) minus Critic Metascore
    # Positive = Web Audience liked it more; Negative = Critics liked it more
    has_both = df["rating_num"].notnull() & df["metascore_num"].notnull()
    df["sentiment_gap"] = np.where(
        has_both,
        ((df["rating_num"] * 10.0) - df["metascore_num"]).round(1),
        np.nan
    )

    # 4. 4-Quadrant Audience vs Critic Matrix
    # Metascore threshold = 60, IMDb Rating threshold = 6.5
    conditions = [
        (df["metascore_num"] >= 60) & (df["rating_num"] >= 6.5),
        (df["metascore_num"] < 60) & (df["rating_num"] >= 6.5),
        (df["metascore_num"] >= 60) & (df["rating_num"] < 6.5),
        (df["metascore_num"] < 60) & (df["rating_num"] < 6.5),
    ]
    choices = [
        "🏆 Universal Hit (Critics & Audience)",
        "🍿 Audience Champion (Cult / Crowd Favorite)",
        "🧐 Critic Darling (Loved by Critics Only)",
        "📉 Universal Miss (Low Critic & Audience)"
    ]
    df["sentiment_quadrant"] = np.select(conditions, choices, default="Unclassified")

    # 5. Engagement Tiers (Quantile-based)
    virality_p95 = df["virality_index"].quantile(0.95)
    virality_p75 = df["virality_index"].quantile(0.75)
    virality_p40 = df["virality_index"].quantile(0.40)

    tier_conditions = [
        df["virality_index"] >= virality_p95,
        df["virality_index"] >= virality_p75,
        df["virality_index"] >= virality_p40,
    ]
    tier_choices = [
        "🔥 Viral Phenomenon (Top 5%)",
        "⚡ High Web Engagement",
        "📊 Steady Interest",
    ]
    df["engagement_tier"] = np.select(tier_conditions, tier_choices, default="🌱 Niche / Long-Tail")

    return df


def format_large_number(num: float) -> str:
    """Formats large numeric totals with readable suffixes (K, M, B)."""
    if pd.isnull(num):
        return "N/A"
    num = float(num)
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return f"{int(num):,}"


def get_web_kpis(df: pd.DataFrame) -> dict:
    """
    Computes key performance indicators for web and audience analytics.
    """
    total_votes = df["votes_num"].sum() if "votes_num" in df.columns else 0
    total_reviews = df["reviews_num"].sum() if "reviews_num" in df.columns else 0
    avg_engagement = df["engagement_rate"].dropna().mean() if "engagement_rate" in df.columns else 0
    avg_virality = df["virality_index"].mean() if "virality_index" in df.columns else 0

    # Critic & audience alignment (titles within 10 pts of each other)
    if "sentiment_gap" in df.columns and df["sentiment_gap"].notnull().any():
        aligned_count = (df["sentiment_gap"].abs() <= 10).sum()
        total_rated = df["sentiment_gap"].notnull().sum()
        alignment_rate = round((aligned_count / total_rated) * 100, 1) if total_rated > 0 else 0
    else:
        alignment_rate = 0

    return {
        "total_votes_raw": int(total_votes),
        "total_votes_str": format_large_number(total_votes),
        "total_reviews_raw": int(total_reviews),
        "total_reviews_str": format_large_number(total_reviews),
        "avg_engagement_rate": round(float(avg_engagement), 2),
        "avg_virality_index": round(float(avg_virality), 1),
        "alignment_rate": alignment_rate,
        "total_movies": len(df)
    }


def get_top_viral_movies(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Returns top movies by Web Virality Index."""
    cols = ["title", "year_num", "rating_num", "metascore_num", "votes_num", "reviews_num", "virality_index", "engagement_tier"]
    avail_cols = [c for c in cols if c in df.columns]
    return (
        df.sort_values("virality_index", ascending=False)
        .head(top_n)[avail_cols]
        .rename(columns={
            "title": "Title",
            "year_num": "Year",
            "rating_num": "IMDb Rating ⭐",
            "metascore_num": "Metascore",
            "votes_num": "Total Votes",
            "reviews_num": "User Reviews",
            "virality_index": "Virality Index 🔥",
            "engagement_tier": "Tier"
        })
    )


def get_audience_vs_critics_outliers(df: pd.DataFrame, top_n: int = 10):
    """
    Identifies titles where web audience loved the movie but critics panned it,
    and titles where critics praised it but web audience disliked it.
    """
    valid = df.dropna(subset=["sentiment_gap", "rating_num", "metascore_num"]).copy()
    # Filter for movies with meaningful votes
    if "votes_num" in valid.columns:
        valid = valid[valid["votes_num"] >= 1000]

    cols = ["title", "year_num", "rating_num", "metascore_num", "sentiment_gap", "votes_num"]
    avail = [c for c in cols if c in valid.columns]

    audience_champions = (
        valid.sort_values("sentiment_gap", ascending=False)
        .head(top_n)[avail]
        .rename(columns={
            "title": "Title",
            "year_num": "Year",
            "rating_num": "IMDb Rating ⭐",
            "metascore_num": "Metascore",
            "sentiment_gap": "Audience Delta (+)",
            "votes_num": "Votes"
        })
    )

    critic_darlings = (
        valid.sort_values("sentiment_gap", ascending=True)
        .head(top_n)[avail]
        .rename(columns={
            "title": "Title",
            "year_num": "Year",
            "rating_num": "IMDb Rating ⭐",
            "metascore_num": "Metascore",
            "sentiment_gap": "Critic Delta (-)",
            "votes_num": "Votes"
        })
    )

    return audience_champions, critic_darlings


def get_web_funnel_summary(df: pd.DataFrame) -> dict:
    """
    Generates a web engagement funnel breakdown.
    Stages:
      1. Cataloged Titles
      2. High Voting Volume (>10k votes)
      3. Active Discussion (>200 user reviews)
      4. Web Phenomenon (Virality Index >= 80)
    """
    total = len(df)
    voted_10k = (df["votes_num"] >= 10000).sum() if "votes_num" in df.columns else 0
    reviewed_200 = (df["reviews_num"] >= 200).sum() if "reviews_num" in df.columns else 0
    viral_phenom = (df["virality_index"] >= 80).sum() if "virality_index" in df.columns else 0

    return {
        "Cataloged Titles": total,
        "High Web Votes (>=10K)": int(voted_10k),
        "Active User Reviews (>=200)": int(reviewed_200),
        "Viral Phenomenon (Index >=80)": int(viral_phenom)
    }
