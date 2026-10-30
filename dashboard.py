"""
🎬 IMDb Comprehensive Analytics & Intelligence Dashboard
Featuring:
  - Interactive Network Graph Analytics (PyVis & Plotly)
  - 'Six Degrees of Separation' Kevin Bacon Actor Path Finder
  - Web & Audience Analytics (Virality Index, Engagement Rate, 4-Quadrant Sentiment Divergence)
  - Genre & Cast Deep-Dives
  - Historical & Multi-Variable Trends
  - Advanced Filtering & Smart Data Export
"""

import ast
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from graph_analytics import (
    build_actor_graph,
    find_actor_path,
    get_ego_network,
    get_top_connected_network,
    compute_network_centrality,
    build_genre_network,
    build_director_actor_network,
    generate_pyvis_html,
    generate_plotly_network,
    parse_list_col
)
from web_analytics import (
    enrich_web_metrics,
    get_web_kpis,
    get_top_viral_movies,
    get_audience_vs_critics_outliers,
    get_web_funnel_summary,
    format_large_number
)

# ---------------------------------------------------------
# App Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="IMDb Network & Web Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern Cinematic CSS Styling
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .imdb-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Custom dark theme adjustments */
.main {
    background-color: #0b0f19;
}

/* Metric Card Container */
.metric-container {
    background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
    border: 1px solid rgba(245, 197, 24, 0.2);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
    margin-bottom: 15px;
}
.metric-container:hover {
    transform: translateY(-2px);
    border-color: rgba(245, 197, 24, 0.55);
}

.metric-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.1;
}

.metric-sub {
    font-size: 0.78rem;
    color: #38bdf8;
    margin-top: 5px;
    font-weight: 500;
}

/* Highlight badge */
.badge-gold {
    background-color: #F5C518;
    color: #000000;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.75rem;
    display: inline-block;
}

.badge-cyan {
    background-color: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.75rem;
    display: inline-block;
}

/* Roadmap step box */
.path-step-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

/* Streamlit tabs override */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #161b22;
    border-radius: 6px 6px 0px 0px;
    padding: 8px 16px;
    color: #94a3b8;
    border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
    background-color: #1f2937 !important;
    color: #F5C518 !important;
    border-bottom: 2px solid #F5C518 !important;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loading & In-Memory Graph Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing IMDb datasets...")
def load_and_prepare_data():
    df_clean = pd.read_csv("imdb_clean.csv")
    df_cast = pd.read_csv("imdb_cast_exploded.csv")
    df_genre = pd.read_csv("imdb_genre_exploded.csv")

    # Column normalization
    for df in [df_clean, df_cast, df_genre]:
        df.columns = [
            c.lower().strip().replace(" ", "_").replace("\n", "_").replace("\t", "_")
            for c in df.columns
        ]

    # Enrich df_clean with Web Analytics & numeric columns
    df_clean = enrich_web_metrics(df_clean)

    # Coerce numeric fields in exploded sets as well
    for df in [df_cast, df_genre]:
        if "rating" in df.columns:
            df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
        if "metascore" in df.columns:
            df["metascore_num"] = pd.to_numeric(df["metascore"], errors="coerce")
            df["metadata"] = df["metascore_num"]
        elif "metadata" in df.columns:
            df["metascore_num"] = pd.to_numeric(df["metadata"], errors="coerce")
        if "duration" in df.columns:
            df["duration_num"] = pd.to_numeric(df["duration"], errors="coerce")
        if "year" in df.columns:
            df["year_num"] = pd.to_numeric(df["year"], errors="coerce")

    return df_clean, df_cast, df_genre


@st.cache_resource(show_spinner="Generating Actor Collaboration Network Graph...")
def get_cached_actor_graph(df: pd.DataFrame):
    return build_actor_graph(df)


@st.cache_resource(show_spinner="Building Genre Co-occurrence Network...")
def get_cached_genre_graph(df: pd.DataFrame):
    return build_genre_network(df)


df_clean, df_cast, df_genre = load_and_prepare_data()
G_actor = get_cached_actor_graph(df_clean)
G_genre = get_cached_genre_graph(df_clean)

# Title column reference
title_col = "title" if "title" in df_clean.columns else None

# ---------------------------------------------------------
# Sidebar: Advanced Interactive Filters
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 12px;'>
        <span style='font-size: 1.8rem;'>🎬</span>
        <span style='font-family: "Outfit", sans-serif; font-size: 1.25rem; font-weight: 800; color: #F5C518;'>
            IMDb ANALYTICS
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.caption("Filter 10,000+ movies by genre, cast, director, year, ratings, and digital popularity.")

# 1. Genre filter
genres_available = sorted([g for g in df_genre["genre"].dropna().unique()]) if "genre" in df_genre.columns else []
selected_genres = st.sidebar.multiselect(
    "🎭 Genres",
    options=genres_available,
    default=[],
    help="Filter movies having ANY of the selected genres."
)

# 2. Cast filter
cast_available = sorted([c for c in df_cast["cast"].dropna().unique()]) if "cast" in df_cast.columns else []
selected_cast = st.sidebar.multiselect(
    "⭐ Cast Members",
    options=cast_available,
    default=[],
    help="Search and filter movies featuring specific actors."
)

# 3. Director filter
directors_available = sorted([d for d in df_clean["director"].dropna().unique() if str(d).strip() != "Unknown"])
selected_directors = st.sidebar.multiselect(
    "🎥 Directors",
    options=directors_available,
    default=[],
    help="Filter by movie director."
)

# 4. Certificate filter
certificates_available = sorted([c for c in df_clean["certificate"].dropna().unique() if str(c).strip() != "Unknown"])
selected_certificates = st.sidebar.multiselect(
    "🏷️ Certificates",
    options=certificates_available,
    default=[],
    help="Age rating / content certificate."
)

st.sidebar.markdown("---")

# 5. Year Range
min_year = int(df_clean["year_num"].dropna().min()) if "year_num" in df_clean.columns else 1920
max_year = int(df_clean["year_num"].dropna().max()) if "year_num" in df_clean.columns else 2024
year_range = st.sidebar.slider("📅 Release Year Range", min_year, max_year, (min_year, max_year))

# 6. IMDb Rating Range
min_r, max_r = 1.0, 10.0
rating_range = st.sidebar.slider("⭐ IMDb Rating Range", min_r, max_r, (min_r, max_r), step=0.1)

# 7. Metascore Range
min_m, max_m = 0, 100
metascore_range = st.sidebar.slider("🎯 Metascore Range", min_m, max_m, (min_m, max_m), step=1)

# 8. Minimum Votes slider
min_votes = st.sidebar.select_slider(
    "🗳️ Minimum Web Votes",
    options=[0, 500, 2000, 10000, 50000, 100000, 250000],
    value=0,
    help="Filter out titles with fewer audience votes."
)

# ---------------------------------------------------------
# Apply Filters
# ---------------------------------------------------------
df_filtered = df_clean.copy()

# Year
if "year_num" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["year_num"].between(year_range[0], year_range[1]) | df_filtered["year_num"].isna()
    ]

# Rating
if "rating_num" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["rating_num"].between(rating_range[0], rating_range[1]) | df_filtered["rating_num"].isna()
    ]

# Metascore
if "metascore_num" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["metascore_num"].between(metascore_range[0], metascore_range[1]) | df_filtered["metascore_num"].isna()
    ]

# Votes
if min_votes > 0 and "votes_num" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["votes_num"] >= min_votes]

# Director
if selected_directors:
    df_filtered = df_filtered[df_filtered["director"].isin(selected_directors)]

# Certificate
if selected_certificates:
    df_filtered = df_filtered[df_filtered["certificate"].isin(selected_certificates)]

# Genre (via df_genre titles)
if selected_genres and title_col and "genre" in df_genre.columns:
    titles_with_genre = df_genre[df_genre["genre"].isin(selected_genres)][title_col].unique().tolist()
    df_filtered = df_filtered[df_filtered[title_col].isin(titles_with_genre)]

# Cast (via df_cast titles)
if selected_cast and title_col and "cast" in df_cast.columns:
    titles_with_cast = df_cast[df_cast["cast"].isin(selected_cast)][title_col].unique().tolist()
    df_filtered = df_filtered[df_filtered[title_col].isin(titles_with_cast)]

# Sidebar Status & Export
pct_catalog = round((len(df_filtered) / len(df_clean)) * 100, 1) if len(df_clean) > 0 else 0
st.sidebar.markdown(
    f"""
    <div style='background: #161b22; border-radius: 8px; padding: 10px 14px; margin: 15px 0;'>
        <div style='color: #94a3b8; font-size: 0.8rem;'>MATCHING MOVIES</div>
        <div style='font-size: 1.4rem; font-weight: 700; color: #F5C518;'>{len(df_filtered):,} <span style='font-size: 0.85rem; color: #64748b;'>({pct_catalog}% of total)</span></div>
    </div>
    """,
    unsafe_allow_html=True
)

csv_download = df_filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "📥 Download Filtered Data (CSV)",
    data=csv_download,
    file_name="imdb_filtered_data.csv",
    mime="text/csv",
    use_container_width=True
)

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown(
    """
    <div style='margin-bottom: 20px;'>
        <h1 style='color: #F5C518; margin-bottom: 4px;'>🎬 IMDb Comprehensive Analytics & Graph Intelligence</h1>
        <p style='color: #94a3b8; font-size: 1.05rem; margin-top: 0;'>
            Multi-dimensional cinematic intelligence combining <b>Network Graph Analytics</b>, <b>Digital Web Analytics</b>, and <b>Deep Exploratory Data Mining</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Overview & KPIs",
    "🕸️ Network Graph Analytics",
    "🌐 Web & Audience Analytics",
    "🎭 Genre Deep-Dive",
    "⭐ Cast & Director Intel",
    "📈 Yearly Trends",
    "🔍 Multi-Variable Plots",
    "🧮 Correlation Heatmap",
    "📑 Data Explorer & Export"
])

# ---------------------------------------------------------
# TAB 1: Overview & Executive KPIs
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Key Executive Metrics")
    kpi_web = get_web_kpis(df_filtered)

    avg_rating_val = round(df_filtered["rating_num"].dropna().mean(), 2) if "rating_num" in df_filtered.columns and df_filtered["rating_num"].notnull().any() else "N/A"
    avg_metascore_val = round(df_filtered["metascore_num"].dropna().mean(), 1) if "metascore_num" in df_filtered.columns and df_filtered["metascore_num"].notnull().any() else "N/A"
    avg_duration_val = int(df_filtered["duration_num"].dropna().mean()) if "duration_num" in df_filtered.columns and df_filtered["duration_num"].notnull().any() else "N/A"

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Catalog Matches</div>
                <div class="metric-value">{len(df_filtered):,}</div>
                <div class="metric-sub">{pct_catalog}% of library</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Avg IMDb Rating</div>
                <div class="metric-value" style="color: #F5C518;">{avg_rating_val} <span style="font-size: 1.2rem;">⭐</span></div>
                <div class="metric-sub">Audience score (1-10)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Avg Metascore</div>
                <div class="metric-value" style="color: #10B981;">{avg_metascore_val}</div>
                <div class="metric-sub">Critic consensus (1-100)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Avg Runtime</div>
                <div class="metric-value">{avg_duration_val} <span style="font-size: 1.1rem; color: #94a3b8;">min</span></div>
                <div class="metric-sub">Feature film length</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Digital Votes</div>
                <div class="metric-value" style="color: #38BDF8;">{kpi_web['total_votes_str']}</div>
                <div class="metric-sub">Total audience votes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c6:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Web Virality</div>
                <div class="metric-value" style="color: #EC4899;">{kpi_web['avg_virality_index']}</div>
                <div class="metric-sub">Avg virality index</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # High-level Leaderboards
    col_lead1, col_lead2 = st.columns(2)

    with col_lead1:
        st.markdown("### 🏆 Top Rated Masterpieces (Min 10K Votes)")
        top_rated = (
            df_filtered[df_filtered["votes_num"] >= 10000]
            .sort_values("rating_num", ascending=False)
            .head(6)[["title", "year_num", "rating_num", "metascore_num", "votes_num", "director"]]
            .rename(columns={
                "title": "Title",
                "year_num": "Year",
                "rating_num": "Rating ⭐",
                "metascore_num": "Metascore",
                "votes_num": "Votes",
                "director": "Director"
            })
        )
        st.dataframe(top_rated, use_container_width=True, hide_index=True)

    with col_lead2:
        st.markdown("### 🔥 Top Viral Web Sensations")
        top_viral = get_top_viral_movies(df_filtered, top_n=6)
        st.dataframe(top_viral[["Title", "Year", "IMDb Rating ⭐", "Total Votes", "Virality Index 🔥"]], use_container_width=True, hide_index=True)

    st.markdown("---")

    # Macro Genre & Cast distributions
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### 🎭 Top Genres Distribution")
        if title_col and "genre" in df_genre.columns and title_col in df_genre.columns:
            g_counts = (
                df_genre[df_genre[title_col].isin(df_filtered[title_col])]
                ["genre"].value_counts().head(12)
            )
            fig_genre_bar = px.bar(
                x=g_counts.values,
                y=g_counts.index,
                orientation="h",
                labels={"x": "Movie Count", "y": "Genre"},
                color=g_counts.values,
                color_continuous_scale="Viridis",
                title="Top Genres in Filtered Dataset"
            )
            fig_genre_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_genre_bar, use_container_width=True)

    with col_g2:
        st.markdown("### ⭐ Top Prolific Cast Members")
        if title_col and "cast" in df_cast.columns and title_col in df_cast.columns:
            c_counts = (
                df_cast[df_cast[title_col].isin(df_filtered[title_col])]
                ["cast"].value_counts().head(12)
            )
            fig_cast_bar = px.bar(
                x=c_counts.values,
                y=c_counts.index,
                orientation="h",
                labels={"x": "Appearances", "y": "Actor"},
                color=c_counts.values,
                color_continuous_scale="Plasma",
                title="Top Actors in Filtered Dataset"
            )
            fig_cast_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cast_bar, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: Network Graph Analytics
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("🕸️ Network Graph Analytics & Hollywood Connectivity")
    st.markdown(
        """
        Uncover the hidden collaborative structure of cinema using **Graph Theory** and **NetworkX**.
        Explore co-starring networks, find degrees of separation between any two actors, and examine Hollywood centrality hubs.
        """
    )

    graph_mode = st.radio(
        "Select Graph Analysis Module:",
        [
            "🌟 Actor Collaboration Network",
            "🔗 Six Degrees of Separation (Path Finder)",
            "🎭 Genre Co-occurrence Network",
            "🎬 Director-Actor Collaboration Web",
            "📊 Network Centrality & Influence Rankings"
        ],
        horizontal=True
    )

    st.markdown("---")

    # 1. Actor Collaboration Network
    if graph_mode == "🌟 Actor Collaboration Network":
        sub_col1, sub_col2, sub_col3 = st.columns([2, 1, 1])

        with sub_col1:
            all_actors_sorted = sorted(list(G_actor.nodes()))
            default_actor = "Leonardo DiCaprio" if "Leonardo DiCaprio" in G_actor else (all_actors_sorted[0] if all_actors_sorted else "")
            focal_actor = st.selectbox(
                "Select Focal Actor (Ego Network):",
                options=["[Show Top Connected Hollywood Network]"] + all_actors_sorted,
                index=1 if default_actor else 0
            )

        with sub_col2:
            viz_engine = st.radio("Visualization Engine:", ["⚡ Interactive PyVis (Physics)", "📊 Plotly 2D"], horizontal=True)

        with sub_col3:
            max_nodes = st.slider("Max Nodes to Display:", 15, 60, 35)

        # Build Subgraph
        if focal_actor == "[Show Top Connected Hollywood Network]":
            sub_g = get_top_connected_network(G_actor, top_n=max_nodes)
            focal_target = None
            title_text = f"Top {max_nodes} Most Connected Actors in Hollywood"
        else:
            sub_g = get_ego_network(G_actor, focal_actor, max_neighbors=max_nodes)
            focal_target = focal_actor
            title_text = f"Co-Starring Ego Network for {focal_actor} ({sub_g.number_of_nodes()} actors)"

        st.caption(f"Showing network with **{sub_g.number_of_nodes()} actors** and **{sub_g.number_of_edges()} co-starring connections**.")

        if viz_engine == "⚡ Interactive PyVis (Physics)":
            html_net = generate_pyvis_html(sub_g, focal_node=focal_target, height="600px")
            st.components.v1.html(html_net, height=620, scrolling=False)
            st.caption("💡 *Tip: Nodes can be dragged and zoomed. Hover over actors to view movie count and average ratings.*")
        else:
            fig_net = generate_plotly_network(sub_g, highlight_nodes=[focal_target] if focal_target else None, title=title_text)
            st.plotly_chart(fig_net, use_container_width=True)

    # 2. Six Degrees of Separation
    elif graph_mode == "🔗 Six Degrees of Separation (Path Finder)":
        st.markdown("### 🔗 The 'Six Degrees of Kevin Bacon' Shortest Path Finder")
        st.markdown("Find the shortest chain of co-starring movie connections connecting any two actors across film history.")

        actors_list = sorted(list(G_actor.nodes()))

        path_c1, path_c2 = st.columns(2)
        with path_c1:
            actor_source = st.selectbox(
                "Origin Actor (Actor A):",
                options=actors_list,
                index=actors_list.index("Leonardo DiCaprio") if "Leonardo DiCaprio" in actors_list else 0
            )
        with path_c2:
            actor_target = st.selectbox(
                "Destination Actor (Actor B):",
                options=actors_list,
                index=actors_list.index("Tom Cruise") if "Tom Cruise" in actors_list else (len(actors_list) - 1)
            )

        if st.button("🚀 Find Shortest Connection Path", type="primary"):
            found, path, connections, path_subgraph = find_actor_path(G_actor, actor_source, actor_target)

            if not found or not path:
                st.warning(f"No direct or indirect collaborative connection found between **{actor_source}** and **{actor_target}** in this dataset.")
            else:
                deg_sep = len(path) - 1
                st.success(f"🎉 Connection Discovered! **{deg_sep} Degree{'s' if deg_sep != 1 else ''} of Separation** ({len(path)} actors in chain).")

                # Visual roadmap
                st.markdown("#### 🛣️ Connection Roadmap:")
                for idx, conn in enumerate(connections):
                    u = conn["from"]
                    v = conn["to"]
                    movies_shared = ", ".join([f"*{m}*" for m in conn["movies"][:3]])
                    st.markdown(
                        f"""
                        <div class="path-step-card">
                            <span class="badge-gold">Step {idx + 1}</span>
                            <b style="color: #f8fafc; font-size: 1.05rem; margin-left: 8px;">{u}</b>
                            <span style="color: #38bdf8;"> ➔ co-starred with ➔ </span>
                            <b style="color: #f8fafc; font-size: 1.05rem;">{v}</b>
                            <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 4px;">
                                🎬 Shared Title(s): {movies_shared}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("#### 🕸️ Path Subgraph Visualization:")
                if path_subgraph:
                    html_path = generate_pyvis_html(path_subgraph, highlight_nodes=path, height="500px")
                    st.components.v1.html(html_path, height=520, scrolling=False)

    # 3. Genre Co-occurrence Network
    elif graph_mode == "🎭 Genre Co-occurrence Network":
        st.markdown("### 🎭 Genre Structural Synergy Network")
        st.markdown("Visualizes how genres co-occur across movies. Thicker edges denote genre combinations that frequently appear together (e.g., Action + Adventure, Comedy + Romance).")

        html_genre = generate_pyvis_html(G_genre, height="600px")
        st.components.v1.html(html_genre, height=620, scrolling=False)

    # 4. Director-Actor Collaboration Web
    elif graph_mode == "🎬 Director-Actor Collaboration Web":
        st.markdown("### 🎬 Director-Actor Collaboration Web")
        st.markdown("Examines the partnerships between top directors and recurring leading actors.")

        top_dirs_count = st.slider("Number of Top Directors to Include:", 5, 25, 12)
        G_dir = build_director_actor_network(df_clean, top_n_directors=top_dirs_count)

        html_dir = generate_pyvis_html(G_dir, height="600px")
        st.components.v1.html(html_dir, height=620, scrolling=False)
        st.caption("🟡 Yellow Nodes = Directors | 🔵 Blue Nodes = Actors")

    # 5. Network Centrality & Hollywood Influence
    elif graph_mode == "📊 Network Centrality & Influence Rankings":
        st.markdown("### 📊 Network Centrality & Hollywood Power Rankings")
        st.markdown(
            """
            - **Degree Centrality**: Most interconnected actors (highest number of distinct co-stars).
            - **PageRank**: Network authority / prestige (connected to other well-connected actors).
            - **Betweenness Centrality**: Key structural bridge actors connecting disparate cliques and genres.
            """
        )
        cent_df = compute_network_centrality(G_actor, top_n=25)
        st.dataframe(cent_df, use_container_width=True, hide_index=True)

        fig_cent = px.bar(
            cent_df.head(15),
            x="Actor",
            y="Degree",
            color="Avg Rating",
            color_continuous_scale="Viridis",
            title="Top 15 Most Connected Actors by Network Degree"
        )
        fig_cent.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_cent, use_container_width=True)


# ---------------------------------------------------------
# TAB 3: Web & Audience Analytics
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("🌐 Digital Web & Audience Engagement Analytics")
    st.markdown(
        """
        Analyze how audiences engage with movies online, examine vote-to-review conversion rates,
        explore the **Web Virality Index**, and discover critic-vs-audience sentiment discrepancies.
        """
    )

    kpis_w = get_web_kpis(df_filtered)

    wk1, wk2, wk3, wk4 = st.columns(4)
    with wk1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Audience Votes Logged</div>
                <div class="metric-value" style="color: #38bdf8;">{kpis_w['total_votes_str']}</div>
                <div class="metric-sub">Raw: {kpis_w['total_votes_raw']:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with wk2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">User Reviews Written</div>
                <div class="metric-value" style="color: #a855f7;">{kpis_w['total_reviews_str']}</div>
                <div class="metric-sub">Raw: {kpis_w['total_reviews_raw']:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with wk3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Review Conversion Rate</div>
                <div class="metric-value" style="color: #10b981;">{kpis_w['avg_engagement_rate']}%</div>
                <div class="metric-sub">Reviews per 100 votes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with wk4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-title">Critic-Audience Consensus</div>
                <div class="metric-value" style="color: #F5C518;">{kpis_w['alignment_rate']}%</div>
                <div class="metric-sub">Within ±10 pts alignment</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 4-Quadrant Sentiment Matrix
    st.markdown("### 🎯 Critic vs Web Audience Sentiment Matrix (4-Quadrant Model)")
    st.caption("X-Axis: Critic Metascore (0-100) | Y-Axis: IMDb Web Audience Rating (scaled 0-100). Quadrant lines at Metascore=60 and Audience=65.")

    valid_sentiment = df_filtered.dropna(subset=["metascore_num", "rating_num"]).copy()
    valid_sentiment["rating_100"] = valid_sentiment["rating_num"] * 10.0

    quad_filter = st.selectbox(
        "Filter by Quadrant Category:",
        options=["All Quadrants"] + sorted(list(valid_sentiment["sentiment_quadrant"].unique()))
    )

    if quad_filter != "All Quadrants":
        chart_data = valid_sentiment[valid_sentiment["sentiment_quadrant"] == quad_filter]
    else:
        chart_data = valid_sentiment

    fig_quad = px.scatter(
        chart_data,
        x="metascore_num",
        y="rating_100",
        color="sentiment_quadrant",
        hover_name="title",
        hover_data=["year_num", "director", "votes_num", "sentiment_gap"],
        labels={
            "metascore_num": "Critic Metascore",
            "rating_100": "IMDb Audience Rating (0-100 Scale)",
            "sentiment_quadrant": "Classification"
        },
        title=f"Audience vs Critic Sentiment Matrix ({len(chart_data):,} movies)",
        opacity=0.75
    )

    # Add quadrant reference lines & shapes
    fig_quad.add_hline(y=65, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)")
    fig_quad.add_vline(x=60, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)")

    fig_quad.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_quad, use_container_width=True)

    # Sentiment Outliers
    st.markdown("#### ⚡ Greatest Sentiment Divergences")
    outlier_col1, outlier_col2 = st.columns(2)

    champs, darlings = get_audience_vs_critics_outliers(df_filtered, top_n=8)

    with outlier_col1:
        st.markdown("🍿 **Top Audience Champions** *(Loved by Web, Panned by Critics)*")
        st.dataframe(champs, use_container_width=True, hide_index=True)

    with outlier_col2:
        st.markdown("🧐 **Top Critic Darlings** *(Praised by Critics, Disliked by Web)*")
        st.dataframe(darlings, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Web Funnel & Virality Breakdown
    w_row1, w_row2 = st.columns(2)

    with w_row1:
        st.markdown("###  funnel-icon 🏁 Web Voting & Engagement Funnel")
        funnel_dict = get_web_funnel_summary(df_filtered)
        fig_funnel = go.Figure(go.Funnel(
            y=list(funnel_dict.keys()),
            x=list(funnel_dict.values()),
            textinfo="value+percent initial",
            marker=dict(color=["#38bdf8", "#818cf8", "#a855f7", "#ec4899"])
        ))
        fig_funnel.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            title="Digital Audience Engagement Funnel"
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with w_row2:
        st.markdown("### 🔥 Top 15 Web Virality Champions")
        top_viral_df = get_top_viral_movies(df_filtered, top_n=12)
        fig_viral_bar = px.bar(
            top_viral_df,
            x="Virality Index 🔥",
            y="Title",
            orientation="h",
            color="Virality Index 🔥",
            color_continuous_scale="Plasma",
            title="Top Titles by Web Virality Index"
        )
        fig_viral_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_viral_bar, use_container_width=True)

    st.markdown("---")

    # Decade Digital Evolution
    st.markdown("### 📅 Digital Age Explosion: Votes & Reviews Growth by Decade")
    if "decade" in df_filtered.columns and "votes_num" in df_filtered.columns:
        decade_stats = (
            df_filtered[df_filtered["decade"] != "Unknown"]
            .groupby("decade")
            .agg(
                Total_Votes=("votes_num", "sum"),
                Total_Reviews=("reviews_num", "sum"),
                Avg_Rating=("rating_num", "mean"),
                Movie_Count=("title", "count")
            )
            .reset_index()
            .sort_values("decade")
        )

        fig_decade_line = go.Figure()
        fig_decade_line.add_trace(go.Bar(
            x=decade_stats["decade"],
            y=decade_stats["Total_Votes"],
            name="Total Web Votes",
            marker_color="#38bdf8",
            yaxis="y"
        ))
        fig_decade_line.add_trace(go.Scatter(
            x=decade_stats["decade"],
            y=decade_stats["Total_Reviews"],
            name="Total User Reviews",
            marker_color="#F5C518",
            yaxis="y2",
            mode="lines+markers"
        ))
        fig_decade_line.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            title="Decade Evolution of Web Voting & Review Volume",
            yaxis=dict(title="Total Votes", side="left"),
            yaxis2=dict(title="Total Reviews", side="right", overlaying="y"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_decade_line, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: Genre Deep-Dive
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("🎭 Comprehensive Genre Analytics")
    if title_col and "genre" in df_genre.columns and title_col in df_genre.columns:
        dfg = df_genre[df_genre[title_col].isin(df_filtered[title_col])].copy()

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            # Frequency
            top_g = dfg["genre"].value_counts().head(20)
            fig_g_cnt = px.bar(
                x=top_g.values,
                y=top_g.index,
                orientation="h",
                labels={"x": "Count", "y": "Genre"},
                title="Top Genres by Frequency",
                color=top_g.values,
                color_continuous_scale="Viridis"
            )
            fig_g_cnt.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_g_cnt, use_container_width=True)

        with g_col2:
            # Average Ratings by Genre
            if "rating_num" in dfg.columns and dfg["rating_num"].notnull().any():
                g_ratings = dfg.groupby("genre")["rating_num"].mean().sort_values(ascending=False).head(20)
                fig_g_rt = px.bar(
                    x=g_ratings.values,
                    y=g_ratings.index,
                    orientation="h",
                    labels={"x": "Avg Rating ⭐", "y": "Genre"},
                    title="Average IMDb Rating by Genre (Top 20)",
                    color=g_ratings.values,
                    color_continuous_scale="Plasma"
                )
                fig_g_rt.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
                st.plotly_chart(fig_g_rt, use_container_width=True)

        st.markdown("---")

        # Boxplots across top 12 genres
        top12_genres = dfg["genre"].value_counts().head(12).index
        dfg_top12 = dfg[dfg["genre"].isin(top12_genres)]

        box_col1, box_col2 = st.columns(2)
        with box_col1:
            if "duration_num" in dfg_top12.columns:
                fig_box_dur = px.box(
                    dfg_top12,
                    x="genre",
                    y="duration_num",
                    title="Duration Distribution across Top 12 Genres",
                    color="genre"
                )
                fig_box_dur.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", showlegend=False)
                st.plotly_chart(fig_box_dur, use_container_width=True)

        with box_col2:
            if "rating_num" in dfg_top12.columns:
                fig_box_rt = px.box(
                    dfg_top12,
                    x="genre",
                    y="rating_num",
                    title="IMDb Rating Distribution across Top 12 Genres",
                    color="genre"
                )
                fig_box_rt.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", showlegend=False)
                st.plotly_chart(fig_box_rt, use_container_width=True)

        # Genre Popularity Heatmap over Years
        if "year_num" in dfg.columns and dfg["year_num"].notnull().any():
            st.markdown("#### 📅 Genre Releases over Years (Heatmap)")
            pivot_g = dfg[dfg["genre"].isin(top12_genres)].pivot_table(
                index="genre",
                columns="year_num",
                values=title_col,
                aggfunc="count"
            ).fillna(0)

            fig_heat_g = px.imshow(
                pivot_g,
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Genre Production Volume across Years"
            )
            fig_heat_g.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig_heat_g, use_container_width=True)
    else:
        st.info("Genre column unavailable.")


# ---------------------------------------------------------
# TAB 5: Cast & Director Intel
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("⭐ Cast & Director Analytics")
    if title_col and "cast" in df_cast.columns and title_col in df_cast.columns:
        dfc = df_cast[df_cast[title_col].isin(df_filtered[title_col])].copy()

        c_row1, c_row2 = st.columns(2)

        with c_row1:
            top_actors = dfc["cast"].value_counts().head(20)
            fig_top_act = px.bar(
                x=top_actors.values,
                y=top_actors.index,
                orientation="h",
                labels={"x": "Movies", "y": "Actor"},
                title="Top 20 Most Frequent Actors",
                color=top_actors.values,
                color_continuous_scale="Viridis"
            )
            fig_top_act.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_top_act, use_container_width=True)

        with c_row2:
            # Director leaderboard
            top_dirs = (
                df_filtered.groupby("director")
                .agg(
                    Movies=("title", "count"),
                    Avg_Rating=("rating_num", "mean"),
                    Total_Votes=("votes_num", "sum")
                )
                .query("Movies >= 3")
                .sort_values("Avg_Rating", ascending=False)
                .head(20)
                .reset_index()
            )
            top_dirs["Avg_Rating"] = top_dirs["Avg_Rating"].round(2)
            fig_top_dir = px.bar(
                top_dirs,
                x="Avg_Rating",
                y="director",
                orientation="h",
                labels={"Avg_Rating": "Avg Rating ⭐", "director": "Director"},
                title="Top Directors by Average Rating (Min 3 Movies)",
                color="Movies",
                color_continuous_scale="Plasma"
            )
            fig_top_dir.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_top_dir, use_container_width=True)

        st.markdown("---")

        # Actor Career Timeline
        top10_actors = dfc["cast"].value_counts().head(10).index.tolist()
        dfc_top10 = dfc[dfc["cast"].isin(top10_actors) & dfc["year_num"].notnull()]

        if not dfc_top10.empty:
            actor_years = dfc_top10.groupby(["year_num", "cast"]).size().reset_index(name="Appearances")
            fig_act_timeline = px.line(
                actor_years,
                x="year_num",
                y="Appearances",
                color="cast",
                title="Top 10 Actors: Yearly Appearances Evolution",
                markers=True
            )
            fig_act_timeline.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                xaxis=dict(title="Year")
            )
            st.plotly_chart(fig_act_timeline, use_container_width=True)


# ---------------------------------------------------------
# TAB 6: Yearly Trends
# ---------------------------------------------------------
with tabs[5]:
    st.subheader("📈 Yearly & Historical Cinematic Trends")
    if "year_num" in df_filtered.columns and df_filtered["year_num"].notnull().any():
        yearly_summary = (
            df_filtered.groupby("year_num")
            .agg(
                Movie_Count=("title", "count"),
                Avg_Duration=("duration_num", "mean"),
                Avg_Rating=("rating_num", "mean"),
                Avg_Metascore=("metascore_num", "mean"),
                Total_Votes=("votes_num", "sum")
            )
            .reset_index()
            .sort_values("year_num")
        )

        yt1, yt2 = st.columns(2)

        with yt1:
            fig_y_count = px.line(
                yearly_summary,
                x="year_num",
                y="Movie_Count",
                title="Movies Produced Per Year",
                markers=True
            )
            fig_y_count.update_traces(line_color="#F5C518")
            fig_y_count.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig_y_count, use_container_width=True)

        with yt2:
            fig_y_dur = px.line(
                yearly_summary,
                x="year_num",
                y="Avg_Duration",
                title="Average Runtime Evolution (Minutes)",
                markers=True
            )
            fig_y_dur.update_traces(line_color="#38BDF8")
            fig_y_dur.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig_y_dur, use_container_width=True)

        st.markdown("---")

        yt3, yt4 = st.columns(2)

        with yt3:
            fig_y_score = go.Figure()
            fig_y_score.add_trace(go.Scatter(
                x=yearly_summary["year_num"],
                y=yearly_summary["Avg_Rating"],
                name="IMDb Rating (1-10)",
                line=dict(color="#F5C518", width=2)
            ))
            fig_y_score.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                title="Yearly Average IMDb Rating",
                yaxis=dict(title="Rating ⭐")
            )
            st.plotly_chart(fig_y_score, use_container_width=True)

        with yt4:
            fig_y_votes = px.bar(
                yearly_summary,
                x="year_num",
                y="Total_Votes",
                title="Total Audience Votes Captured by Release Year",
                color="Total_Votes",
                color_continuous_scale="Viridis"
            )
            fig_y_votes.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", coloraxis_showscale=False)
            st.plotly_chart(fig_y_votes, use_container_width=True)


# ---------------------------------------------------------
# TAB 7: Multi-Variable Scatter Plots
# ---------------------------------------------------------
with tabs[6]:
    st.subheader("🔍 Multi-Variable Explorations & Scatter Analysis")

    sc1, sc2 = st.columns(2)

    with sc1:
        if {"duration_num", "rating_num"}.issubset(df_filtered.columns):
            fig_sc1 = px.scatter(
                df_filtered,
                x="duration_num",
                y="rating_num",
                color="year_num" if "year_num" in df_filtered.columns else None,
                hover_name="title",
                hover_data=["director", "votes_num"],
                title="IMDb Rating vs Duration (Minutes)",
                labels={"duration_num": "Duration (min)", "rating_num": "Rating ⭐", "year_num": "Year"}
            )
            fig_sc1.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig_sc1, use_container_width=True)

    with sc2:
        if {"votes_num", "rating_num"}.issubset(df_filtered.columns):
            fig_sc2 = px.scatter(
                df_filtered,
                x="votes_num",
                y="rating_num",
                log_x=True,
                color="metascore_num" if "metascore_num" in df_filtered.columns else None,
                hover_name="title",
                hover_data=["year_num", "director"],
                title="IMDb Rating vs Total Votes (Logarithmic Scale)",
                labels={"votes_num": "Total Votes (Log Scale)", "rating_num": "Rating ⭐", "metascore_num": "Metascore"}
            )
            fig_sc2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
            st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown("---")

    # 3D Scatter Option
    if {"rating_num", "metascore_num", "duration_num"}.issubset(df_filtered.columns):
        st.markdown("#### 🌐 3D Cinematic Feature Space")
        df_3d = df_filtered.dropna(subset=["rating_num", "metascore_num", "duration_num"]).head(1500)
        fig_3d = px.scatter_3d(
            df_3d,
            x="duration_num",
            y="metascore_num",
            z="rating_num",
            color="rating_num",
            hover_name="title",
            title="3D Feature Space: Duration vs Metascore vs IMDb Rating",
            labels={"duration_num": "Duration", "metascore_num": "Metascore", "rating_num": "Rating"}
        )
        fig_3d.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", height=600)
        st.plotly_chart(fig_3d, use_container_width=True)


# ---------------------------------------------------------
# TAB 8: Correlation Heatmap
# ---------------------------------------------------------
with tabs[7]:
    st.subheader("🧮 Correlation Analysis")
    st.markdown("Statistical Pearson correlation matrix across continuous numeric and web analytics indicators.")

    corr_candidates = [
        "rating_num", "metascore_num", "duration_num", "year_num",
        "votes_num", "reviews_num", "virality_index", "engagement_rate"
    ]
    num_cols = [c for c in corr_candidates if c in df_filtered.columns and df_filtered[c].dropna().nunique() > 1]

    if len(num_cols) >= 2:
        clean_col_names = [c.replace("_num", "").replace("_", " ").title() for c in num_cols]
        corr_matrix = df_filtered[num_cols].corr().round(2)
        corr_matrix.columns = clean_col_names
        corr_matrix.index = clean_col_names

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Continuous Numeric Feature Correlation Heatmap"
        )
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Insufficient numeric columns in current filter subset for correlation analysis.")


# ---------------------------------------------------------
# TAB 9: Data Explorer & Smart Export
# ---------------------------------------------------------
with tabs[8]:
    st.subheader("📑 Interactive Data Explorer & Export")
    st.caption("Search, sort, filter, and inspect movie-level records.")

    search_query = st.text_input("🔍 Search by Title or Director:", placeholder="Type title or director name...")
    display_df = df_filtered.copy()

    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            display_df["title"].astype(str).str.lower().str.contains(q, na=False) |
            display_df["director"].astype(str).str.lower().str.contains(q, na=False)
        )
        display_df = display_df[mask]

    # Select columns to display
    display_cols = [
        "title", "year_num", "certificate", "duration_num", "rating_num",
        "metascore_num", "director", "votes_num", "reviews_num", "virality_index", "sentiment_quadrant"
    ]
    avail_display = [c for c in display_cols if c in display_df.columns]
    st.dataframe(display_df[avail_display], use_container_width=True, height=550)

    st.markdown(
        f"""
        <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 8px;">
            Showing <b>{len(display_df):,}</b> records matching active filters and query.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        label="📥 Download Displayed Records as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="imdb_explorer_export.csv",
        mime="text/csv",
        key="btn_explorer_export"
    )
