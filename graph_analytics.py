"""
graph_analytics.py
Comprehensive Network Graph Analytics module for IMDb Movie Data.
Provides:
  - Actor co-starring network generation
  - 'Six Degrees of Separation' / shortest path discovery
  - Ego-network extraction and top-tier Hollywood connectivity
  - Genre co-occurrence network
  - Director-Actor collaboration networks
  - Centrality metrics (Degree, PageRank, Betweenness)
  - Interactive PyVis physics rendering & Plotly fallback visualizers
"""

import ast
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network


def parse_list_col(val):
    """Safely parse a stringified python list or list object."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        # Fallback to comma-separated split
        cleaned = val.strip("[]'\" ")
        return [item.strip(" '\"") for item in cleaned.split(",") if item.strip()]


def build_actor_graph(df: pd.DataFrame, max_movies: int = None) -> nx.Graph:
    """
    Builds an undirected co-starring network from movie data.
    Nodes = Actors (with metadata: movie_count, avg_rating, top_titles)
    Edges = Co-starred in a movie (with weight: count, movies: list of titles)
    """
    G = nx.Graph()
    subset = df.head(max_movies) if max_movies else df

    actor_movies = {}
    actor_ratings = {}

    for _, row in subset.iterrows():
        title = str(row.get("title", "Unknown"))
        cast = parse_list_col(row.get("cast", []))
        rating = row.get("rating")
        try:
            rating_val = float(rating) if pd.notnull(rating) else None
        except (ValueError, TypeError):
            rating_val = None

        # Track actor-level stats
        for actor in cast:
            actor = str(actor).strip()
            if not actor:
                continue
            if actor not in actor_movies:
                actor_movies[actor] = []
                actor_ratings[actor] = []
            actor_movies[actor].append(title)
            if rating_val is not None:
                actor_ratings[actor].append(rating_val)

        # Connect all pairs of actors co-starring in this movie
        for i in range(len(cast)):
            actor_a = str(cast[i]).strip()
            if not actor_a:
                continue
            for j in range(i + 1, len(cast)):
                actor_b = str(cast[j]).strip()
                if not actor_b or actor_a == actor_b:
                    continue

                if G.has_edge(actor_a, actor_b):
                    G[actor_a][actor_b]["weight"] += 1
                    if title not in G[actor_a][actor_b]["movies"]:
                        G[actor_a][actor_b]["movies"].append(title)
                else:
                    G.add_edge(actor_a, actor_b, weight=1, movies=[title])

    # Populate node attributes
    for actor in G.nodes():
        m_list = actor_movies.get(actor, [])
        r_list = actor_ratings.get(actor, [])
        avg_r = round(float(np.mean(r_list)), 2) if r_list else 6.5
        G.nodes[actor]["movie_count"] = len(m_list)
        G.nodes[actor]["avg_rating"] = avg_r
        G.nodes[actor]["movies"] = m_list[:10]  # preview

    return G


def find_actor_path(G: nx.Graph, actor_a: str, actor_b: str):
    """
    Finds the shortest collaboration path between two actors ('Six Degrees of Kevin Bacon').
    Returns:
      - found (bool)
      - path (list of actor names)
      - connections (list of dicts with from, to, and shared movies)
      - subgraph (nx.Graph containing path and immediate context)
    """
    if actor_a not in G or actor_b not in G:
        return False, [], [], None

    if actor_a == actor_b:
        return True, [actor_a], [], G.subgraph([actor_a]).copy()

    try:
        path = nx.shortest_path(G, source=actor_a, target=actor_b)
        connections = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            shared_movies = G[u][v].get("movies", [])
            connections.append({
                "from": u,
                "to": v,
                "movies": shared_movies,
                "weight": G[u][v].get("weight", 1)
            })

        # Build path subgraph plus 1-hop neighbors of path for rich context
        path_nodes = set(path)
        context_nodes = set(path)
        for node in path:
            context_nodes.update(list(G.neighbors(node))[:3])  # small context

        subgraph = G.subgraph(context_nodes).copy()
        return True, path, connections, subgraph

    except nx.NetworkXNoPath:
        return False, [], [], None


def get_ego_network(G: nx.Graph, focal_actor: str, max_neighbors: int = 35) -> nx.Graph:
    """
    Extracts the ego network (1st degree and 2nd degree co-stars) for a specific actor.
    """
    if focal_actor not in G:
        return nx.Graph()

    neighbors = list(G.neighbors(focal_actor))
    # Sort neighbors by co-star weight
    neighbors.sort(key=lambda n: G[focal_actor][n].get("weight", 1), reverse=True)
    selected_nodes = [focal_actor] + neighbors[:max_neighbors]

    return G.subgraph(selected_nodes).copy()


def get_top_connected_network(G: nx.Graph, top_n: int = 40, min_weight: int = 1) -> nx.Graph:
    """
    Extracts the subgraph of the most connected actors in the dataset.
    """
    if G.number_of_nodes() == 0:
        return nx.Graph()

    degrees = dict(G.degree())
    top_actors = sorted(degrees.keys(), key=lambda a: degrees[a], reverse=True)[:top_n]
    sub = G.subgraph(top_actors).copy()

    # Filter out edges below min_weight
    if min_weight > 1:
        edges_to_remove = [(u, v) for u, v, d in sub.edges(data=True) if d.get("weight", 1) < min_weight]
        sub.remove_edges_from(edges_to_remove)
        # Drop isolates
        isolates = list(nx.isolates(sub))
        sub.remove_nodes_from(isolates)

    return sub


def compute_network_centrality(G: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    """
    Calculates Degree Centrality, PageRank, and Betweenness Centrality for actors.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    deg_cent = nx.degree_centrality(G)
    pagerank = nx.pagerank(G, max_iter=100, weight="weight")

    # For large graphs, sample betweenness or calculate on largest component for speed
    if G.number_of_nodes() > 1000:
        # Use degree top 200 nodes for betweenness approximation to stay instantaneous
        top_deg_nodes = sorted(deg_cent.keys(), key=lambda k: deg_cent[k], reverse=True)[:250]
        sub = G.subgraph(top_deg_nodes)
        bet_cent = nx.betweenness_centrality(sub, weight="weight")
    else:
        bet_cent = nx.betweenness_centrality(G, weight="weight")

    records = []
    for actor in sorted(deg_cent.keys(), key=lambda a: deg_cent[a], reverse=True)[:top_n]:
        movie_count = G.nodes[actor].get("movie_count", G.degree(actor))
        avg_rating = G.nodes[actor].get("avg_rating", 0.0)
        records.append({
            "Actor": actor,
            "Degree": G.degree(actor),
            "Degree Centrality": round(deg_cent.get(actor, 0), 4),
            "PageRank Influence": round(pagerank.get(actor, 0) * 1000, 2),
            "Betweenness Centrality": round(bet_cent.get(actor, 0), 4),
            "Total Movies": movie_count,
            "Avg Rating": avg_rating
        })

    return pd.DataFrame(records)


def build_genre_network(df: pd.DataFrame) -> nx.Graph:
    """
    Builds a genre co-occurrence network.
    Nodes = Genres
    Edges = Frequency of co-occurrence in the same title
    """
    G = nx.Graph()
    genre_counts = {}

    for _, row in df.iterrows():
        genres = parse_list_col(row.get("genre", []))
        for g in genres:
            g = str(g).strip()
            if not g:
                continue
            genre_counts[g] = genre_counts.get(g, 0) + 1

        for i in range(len(genres)):
            g1 = str(genres[i]).strip()
            if not g1:
                continue
            for j in range(i + 1, len(genres)):
                g2 = str(genres[j]).strip()
                if not g2 or g1 == g2:
                    continue
                if G.has_edge(g1, g2):
                    G[g1][g2]["weight"] += 1
                else:
                    G.add_edge(g1, g2, weight=1)

    for g, count in genre_counts.items():
        if g in G:
            G.nodes[g]["count"] = count

    return G


def build_director_actor_network(df: pd.DataFrame, top_n_directors: int = 15) -> nx.Graph:
    """
    Builds a bipartite collaboration network between top directors and their primary cast.
    """
    G = nx.Graph()
    top_dirs = df["director"].value_counts().head(top_n_directors).index.tolist()
    dir_df = df[df["director"].isin(top_dirs)]

    for _, row in dir_df.iterrows():
        d = str(row.get("director", "")).strip()
        cast = parse_list_col(row.get("cast", []))
        title = str(row.get("title", ""))

        if not d:
            continue

        if not G.has_node(d):
            G.add_node(d, node_type="director", color="#F5C518", size=24)

        for actor in cast[:4]:  # Top billed actors
            actor = str(actor).strip()
            if not actor:
                continue
            if not G.has_node(actor):
                G.add_node(actor, node_type="actor", color="#38BDF8", size=15)

            if G.has_edge(d, actor):
                G[d][actor]["weight"] += 1
                if title not in G[d][actor]["movies"]:
                    G[d][actor]["movies"].append(title)
            else:
                G.add_edge(d, actor, weight=1, movies=[title])

    return G


def generate_pyvis_html(
    subgraph: nx.Graph,
    highlight_nodes: list = None,
    focal_node: str = None,
    height: str = "550px"
) -> str:
    """
    Generates an interactive, dark-themed HTML physics visualization of the network using PyVis.
    """
    if subgraph.number_of_nodes() == 0:
        return "<div style='color: #94a3b8; text-align: center; padding: 40px;'>No graph nodes to display with current filters.</div>"

    net = Network(
        height=height,
        width="100%",
        bgcolor="#0e1117",
        font_color="#f8fafc",
        directed=False
    )

    highlight_set = set(highlight_nodes or [])

    for node in subgraph.nodes():
        node_str = str(node)
        degree = subgraph.degree(node)
        avg_rating = subgraph.nodes[node].get("avg_rating", "N/A")
        movie_count = subgraph.nodes[node].get("movie_count", degree)
        movies_preview = subgraph.nodes[node].get("movies", [])
        custom_color = subgraph.nodes[node].get("color")
        custom_size = subgraph.nodes[node].get("size")

        # Determine styling
        if node_str == focal_node:
            color = "#F5C518"  # IMDb gold
            size = 32
            border_color = "#FFFFFF"
        elif node_str in highlight_set:
            color = "#EC4899"  # Hot pink for path highlight
            size = 26
            border_color = "#F43F5E"
        elif custom_color:
            color = custom_color
            size = custom_size or 18
            border_color = "#334155"
        else:
            # Color by rating or degree
            if isinstance(avg_rating, (int, float)):
                if avg_rating >= 7.5:
                    color = "#10B981"  # Emerald
                elif avg_rating >= 6.5:
                    color = "#38BDF8"  # Cyan
                else:
                    color = "#818CF8"  # Indigo
            else:
                color = "#38BDF8"
            size = max(12, min(30, 12 + degree * 1.5))
            border_color = "#1E293B"

        title_tooltip = (
            f"<div style='font-family: sans-serif; font-size: 13px; line-height: 1.4; color: #1e293b;'>"
            f"<b>{node_str}</b><br/>"
            f"• Connections: {degree}<br/>"
            f"• Movies Count: {movie_count}<br/>"
            f"• Avg Rating: {avg_rating} ⭐<br/>"
        )
        if movies_preview:
            title_tooltip += f"• Notable: {', '.join(movies_preview[:3])}<br/>"
        title_tooltip += "</div>"

        net.add_node(
            node_str,
            label=node_str,
            title=title_tooltip,
            color={"background": color, "border": border_color, "highlight": {"background": "#F5C518", "border": "#FFFFFF"}},
            size=size,
            borderWidth=2,
            font={"size": 13, "color": "#f8fafc", "face": "Inter, sans-serif"}
        )

    for u, v, data in subgraph.edges(data=True):
        u_str, v_str = str(u), str(v)
        weight = data.get("weight", 1)
        movies = data.get("movies", [])
        is_highlight_edge = (highlight_set and u_str in highlight_set and v_str in highlight_set)

        edge_color = "#EC4899" if is_highlight_edge else "#334155"
        edge_width = 3 if is_highlight_edge else max(1, min(6, weight * 1.5))

        edge_title = f"{u_str} & {v_str}: {weight} shared movie(s)"
        if movies:
            edge_title += f"<br/>Films: {', '.join(movies[:4])}"

        net.add_edge(
            u_str,
            v_str,
            value=weight,
            title=edge_title,
            color=edge_color,
            width=edge_width
        )

    # Configure physics for natural layout
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3500,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "stabilization": {"iterations": 80}
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true
      }
    }
    """)

    return net.generate_html()


def generate_plotly_network(
    subgraph: nx.Graph,
    highlight_nodes: list = None,
    title: str = "Network Graph"
) -> go.Figure:
    """
    Creates a responsive 2D Plotly network graph visualization.
    """
    if subgraph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title=title,
            annotations=[dict(text="No nodes to display", showarrow=False, font=dict(size=16, color="#94a3b8"))]
        )
        return fig

    pos = nx.spring_layout(subgraph, k=0.35, iterations=50, seed=42)
    highlight_set = set(highlight_nodes or [])

    # Edge traces
    edge_x = []
    edge_y = []
    for u, v in subgraph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="rgba(100, 116, 139, 0.45)"),
        hoverinfo="none",
        mode="lines"
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []

    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        deg = subgraph.degree(node)
        avg_r = subgraph.nodes[node].get("avg_rating", 6.5)
        m_count = subgraph.nodes[node].get("movie_count", deg)

        tooltip = f"<b>{node}</b><br>Degree: {deg}<br>Movies: {m_count}<br>Avg Rating: {avg_r} ⭐"
        node_text.append(tooltip)

        if node in highlight_set:
            node_sizes.append(28)
            node_colors.append("#EC4899")
        else:
            node_sizes.append(max(10, min(32, 10 + deg * 1.5)))
            node_colors.append(avg_r)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) if subgraph.degree(n) >= 4 or n in highlight_set else "" for n in subgraph.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title=dict(text="Avg Rating ⭐", side="right"),
                xanchor="left"
            ),
            line_width=2,
            line_color="#1e293b"
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=18, color="#F5C518")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117"
        )
    )

    return fig
