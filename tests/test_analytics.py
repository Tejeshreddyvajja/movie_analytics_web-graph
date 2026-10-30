"""
tests/test_analytics.py
Unit tests for graph_analytics and web_analytics modules.
"""

import pandas as pd
import numpy as np
import pytest
import networkx as nx

from graph_analytics import (
    build_actor_graph,
    find_actor_path,
    get_ego_network,
    get_top_connected_network,
    compute_network_centrality,
    build_genre_network,
    build_director_actor_network,
    generate_plotly_network,
)
from web_analytics import (
    enrich_web_metrics,
    get_web_kpis,
    get_top_viral_movies,
    get_audience_vs_critics_outliers,
    get_web_funnel_summary,
    format_large_number
)


@pytest.fixture
def sample_movie_df():
    data = {
        "title": ["Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "Django Unchained"],
        "year": [2010, 2008, 2014, 1994, 2012],
        "rating": [8.8, 9.0, 8.7, 8.9, 8.5],
        "metascore": [74.0, 84.0, 74.0, 95.0, 81.0],
        "duration": [148, 152, 169, 154, 165],
        "votes": ["2,500,000", "2,800,000", "1,900,000", "2,100,000", "1,600,000"],
        "review_count": ["4,100", "7,200", "4,800", "3,500", "2,200"],
        "director": ["Christopher Nolan", "Christopher Nolan", "Christopher Nolan", "Quentin Tarantino", "Quentin Tarantino"],
        "genre": [
            "['Action', 'Adventure', 'Sci-Fi']",
            "['Action', 'Crime', 'Drama']",
            "['Adventure', 'Drama', 'Sci-Fi']",
            "['Crime', 'Drama']",
            "['Drama', 'Western']"
        ],
        "cast": [
            "['Leonardo DiCaprio', 'Joseph Gordon-Levitt', 'Elliot Page', 'Tom Hardy']",
            "['Christian Bale', 'Heath Ledger', 'Aaron Eckhart', 'Michael Caine']",
            "['Matthew McConaughey', 'Anne Hathaway', 'Jessica Chastain', 'Michael Caine']",
            "['John Travolta', 'Uma Thurman', 'Samuel L. Jackson', 'Bruce Willis']",
            "['Jamie Foxx', 'Christoph Waltz', 'Leonardo DiCaprio', 'Samuel L. Jackson']"
        ]
    }
    return pd.DataFrame(data)


def test_enrich_web_metrics(sample_movie_df):
    df_enriched = enrich_web_metrics(sample_movie_df)

    assert "votes_num" in df_enriched.columns
    assert "reviews_num" in df_enriched.columns
    assert "rating_num" in df_enriched.columns
    assert "metascore_num" in df_enriched.columns
    assert "metadata" in df_enriched.columns
    assert "virality_index" in df_enriched.columns
    assert "engagement_rate" in df_enriched.columns
    assert "sentiment_gap" in df_enriched.columns
    assert "sentiment_quadrant" in df_enriched.columns

    # Check numeric conversions
    assert df_enriched.loc[0, "votes_num"] == 2500000
    assert df_enriched.loc[0, "reviews_num"] == 4100
    assert df_enriched.loc[0, "rating_num"] == 8.8
    assert df_enriched.loc[0, "metascore_num"] == 74.0

    # Virality index should be within 0-100
    assert (df_enriched["virality_index"] >= 0).all()
    assert (df_enriched["virality_index"] <= 100).all()

    # Inception: rating 8.8 * 10 - 74 = 88 - 74 = 14
    assert round(df_enriched.loc[0, "sentiment_gap"], 1) == 14.0


def test_web_kpis_and_funnel(sample_movie_df):
    df_enriched = enrich_web_metrics(sample_movie_df)
    kpis = get_web_kpis(df_enriched)

    assert kpis["total_votes_raw"] > 10_000_000
    assert "M" in kpis["total_votes_str"]
    assert kpis["total_movies"] == 5

    funnel = get_web_funnel_summary(df_enriched)
    assert funnel["Cataloged Titles"] == 5
    assert funnel["High Web Votes (>=10K)"] == 5


def test_actor_graph_construction(sample_movie_df):
    G = build_actor_graph(sample_movie_df)

    assert G.number_of_nodes() > 10
    assert G.number_of_edges() > 10

    # Leonardo DiCaprio and Tom Hardy co-starred in Inception
    assert G.has_edge("Leonardo DiCaprio", "Tom Hardy")
    assert "Inception" in G["Leonardo DiCaprio"]["Tom Hardy"]["movies"]

    # Leonardo DiCaprio and Samuel L. Jackson co-starred in Django Unchained
    assert G.has_edge("Leonardo DiCaprio", "Samuel L. Jackson")
    assert "Django Unchained" in G["Leonardo DiCaprio"]["Samuel L. Jackson"]["movies"]


def test_find_actor_path(sample_movie_df):
    G = build_actor_graph(sample_movie_df)

    # Path between Tom Hardy (Inception) and Christoph Waltz (Django)
    # Tom Hardy -> Leonardo DiCaprio -> Christoph Waltz
    found, path, conns, sub = find_actor_path(G, "Tom Hardy", "Christoph Waltz")
    assert found is True
    assert len(path) == 3
    assert path[0] == "Tom Hardy"
    assert path[1] == "Leonardo DiCaprio"
    assert path[2] == "Christoph Waltz"
    assert len(conns) == 2


def test_find_actor_path_missing(sample_movie_df):
    G = build_actor_graph(sample_movie_df)
    found, path, conns, sub = find_actor_path(G, "Unknown Actor X", "Unknown Actor Y")
    assert found is False
    assert path == []


def test_ego_network_and_centrality(sample_movie_df):
    G = build_actor_graph(sample_movie_df)
    ego = get_ego_network(G, "Leonardo DiCaprio", max_neighbors=10)
    assert "Leonardo DiCaprio" in ego.nodes()
    assert ego.number_of_nodes() > 1

    cent = compute_network_centrality(G, top_n=5)
    assert isinstance(cent, pd.DataFrame)
    assert len(cent) == 5
    assert "Actor" in cent.columns
    assert "Degree Centrality" in cent.columns


def test_genre_and_director_networks(sample_movie_df):
    G_genre = build_genre_network(sample_movie_df)
    assert G_genre.has_node("Action")
    assert G_genre.has_node("Crime")
    assert G_genre.has_edge("Action", "Crime")

    G_dir = build_director_actor_network(sample_movie_df, top_n_directors=2)
    assert G_dir.has_node("Christopher Nolan")
    assert G_dir.has_edge("Christopher Nolan", "Leonardo DiCaprio")


def test_format_large_number():
    assert format_large_number(1_250_000) == "1.2M"
    assert format_large_number(5_400_000_000) == "5.40B"
    assert format_large_number(450) == "450"
    assert format_large_number(None) == "N/A"
