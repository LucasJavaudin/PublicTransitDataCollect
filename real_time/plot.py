import os
from collections import defaultdict
from datetime import time
import itertools

import numpy as np
import networkx as nx
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
if not os.path.isdir(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

GRAPH_NAME = "2025-11-13-A.png"

#  LINE = "STIF:Line::C01741:"  # Transilien U.
LINE = "STIF:Line::C01742:"  # RER A.
#  LINE = "STIF:Line::C01384:"  # M14.
#  LINE = "STIF:Line::C01372:"  # M2.
DATE = "2025-11-13"
#  FROM = "Cergy Le Haut"
FROM = "Vincennes"
#  FROM = "Gare de Poissy"
#  TO = "Boissy-Saint-Léger"
#  TO = "Gare de La Verrière"
TO = "Cergy Le Haut"
COLOR = "red"
COLORS = mpl.colormaps["Dark2"]

df = (
    pl.scan_parquet(os.path.join(DATA_DIR, f"{DATE}.parquet"))
    .filter(pl.col("line_ref") == LINE)
    .filter(pl.col("exp_arr_time").is_not_null() | pl.col("exp_dep_time").is_not_null())
    .with_columns(
        pl.col("exp_arr_time")
        .fill_null(pl.col("exp_dep_time"))
        .dt.convert_time_zone("Europe/Paris"),
        pl.col("exp_dep_time")
        .fill_null(pl.col("exp_arr_time"))
        .dt.convert_time_zone("Europe/Paris"),
        pl.col("aim_arr_time")
        .fill_null(pl.col("aim_dep_time"))
        .dt.convert_time_zone("Europe/Paris"),
        pl.col("aim_dep_time")
        .fill_null(pl.col("aim_arr_time"))
        .dt.convert_time_zone("Europe/Paris"),
    )
    .with_columns(
        mean_time=pl.col("exp_arr_time").dt.offset_by(
            pl.format(
                "{}s", (pl.col("exp_dep_time") - pl.col("exp_arr_time")).dt.total_seconds() // 2
            )
        )
    )
    .filter(
        pl.col("mean_time").dt.date().dt.to_string() == DATE,
        pl.col("mean_time").dt.time() > time(hour=4),
    )
    #  .filter(
    #  (
    #  pl.col("exp_dep_time").max().over("journey_ref")
    #  - pl.col("exp_dep_time").min().over("journey_ref")
    #  ).dt.total_seconds()
    #  > 60 * 25
    #  )
)

stops = pl.scan_parquet(os.path.join(BASE_DIR, "idf", "arrets.parquet")).with_columns(
    stop_ref=pl.lit("STIF:StopPoint:Q:") + pl.col("arrid") + pl.lit(":")
)

df = df.join(stops, on="stop_ref")

df = df.sort("aim_arr_time", "aim_dep_time", "mean_time")

#  df = df.with_columns(
#  is_from=(pl.col("arrname").str.to_lowercase() == FROM.lower()).cast(pl.UInt8),
#  is_to=(pl.col("arrname").str.to_lowercase() == TO.lower()).cast(pl.UInt8),
#  )

df = df.group_by("journey_ref").agg(
    "arrname", "mean_time", "exp_arr_time", "exp_dep_time", "aim_arr_time", "aim_dep_time"
)

df = df.collect()

print(f"Number of journey: {len(df):,}")

# Find all stop pairs on the line.
direct_connections_times = defaultdict(lambda: list())
indirect_connections = set()
for journey in df.partition_by("journey_ref"):
    trip_stops = np.array(journey["arrname"][0])
    unique, idx, counts = np.unique(trip_stops, return_counts=True, return_index=True)
    idx.sort()
    uniques = trip_stops[idx]
    arr_times = journey["exp_arr_time"][0][idx]
    dep_times = journey["exp_dep_time"][0][idx]
    looped_stops = set(unique[counts > 1])
    for i, (a, b) in enumerate(itertools.pairwise(uniques)):
        t = arr_times[i + 1] - dep_times[i]
        direct_connections_times[f"{a}->{b}"].append(t.total_seconds())
    indirect_connections |= {
        f"{a}->{b}"
        for i, a in enumerate(trip_stops[:-2])
        for b in trip_stops[i + 2 :]
        if a not in looped_stops and b not in looped_stops
    }

direct_connections = set(direct_connections_times.keys())
connections = direct_connections.difference(indirect_connections)

# Find the path FROM -> TO.
G = nx.DiGraph()
for pair in connections:
    t = np.median(direct_connections_times[pair])
    a, b = pair.split("->")
    G.add_edge(a.lower(), b.lower(), time=t)

# FIXME: Required for RER A.
if LINE == "STIF:Line::C01742:":
    G.add_edge("houilles - carrières-sur-seine", "nanterre - préfecture", time=5 * 60)
    G.add_edge("nanterre - préfecture", "houilles - carrières-sur-seine", time=5 * 60)

# FIXME: Only one direction is added some times.
for a, b in G.edges():
    G.add_edge(b, a, **G.edges[a, b])

# Find all the endpoints (in-degree = 2).
endpoints = list(map(lambda it: it[0], filter(lambda it: it[1] == 1, dict(G.in_degree).items())))

# Find the most "central" node.
edge_btwnss = nx.edge_betweenness_centrality(G)
central_edge = sorted(edge_btwnss.items(), key=lambda it: it[1], reverse=True)[0]
s, t = central_edge[0]
st_str = f"{s}->{t}"
ts_str = f"{t}->{s}"

paths = nx.all_pairs_shortest_path(G)
fwd_paths = list()
bwd_paths = list()
for origin, spaths in paths:
    for destination, path in spaths.items():
        path_str = "->".join(path)
        if st_str in path_str:
            # Forward path.
            fwd_paths.append(path)
        elif ts_str in path_str:
            # Backward path.
            bwd_paths.append(path)

G_fwd = nx.DiGraph()
for path in fwd_paths:
    for i in range(len(path) - 1):
        G_fwd.add_edge(path[i], path[i + 1])
G_bwd = nx.DiGraph()
for path in bwd_paths:
    for i in range(len(path) - 1):
        G_bwd.add_edge(path[i], path[i + 1])

fwd_endpoints = list()
bwd_endpoints = list()
all_fwd_endpoints_nodes = dict()
all_bwd_endpoints_nodes = dict()
for endpoint in endpoints:
    if any(p[-1] == endpoint for p in fwd_paths):
        assert not any(p[-1] == endpoint for p in bwd_paths)
        # This is an endpoint in forward direction.
        fwd_endpoints.append(endpoint)
        all_fwd_endpoints_nodes[endpoint] = set(
            n for p in fwd_paths if p[-1] == endpoint for n in p
        )
    else:
        assert any(p[-1] == endpoint for p in bwd_paths)
        bwd_endpoints.append(endpoint)
        all_bwd_endpoints_nodes[endpoint] = set(
            n for p in bwd_paths if p[-1] == endpoint for n in p
        )


fwd_endpoints_nodes = dict()
bwd_endpoints_nodes = dict()
for endpoint, nodes in all_fwd_endpoints_nodes.items():
    for _, other_nodes in filter(lambda it: it[0] != endpoint, all_fwd_endpoints_nodes.items()):
        nodes = nodes - other_nodes
    fwd_endpoints_nodes[endpoint] = nodes
for endpoint, nodes in all_bwd_endpoints_nodes.items():
    for _, other_nodes in filter(lambda it: it[0] != endpoint, all_bwd_endpoints_nodes.items()):
        nodes = nodes - other_nodes
    bwd_endpoints_nodes[endpoint] = nodes


def add_nodes_from_path(main_path: list[str], path: list[str], G: nx.DiGraph):
    for n0 in path:
        assert n0 not in main_path
        for i, n1 in enumerate(main_path):
            if G.has_edge(n0, n1):
                main_path.insert(i, n0)
                path.remove(n0)
                add_nodes_from_path(main_path, path, G)
                break
            if G.has_edge(n1, n0):
                main_path.insert(i + 1, n0)
                path.remove(n0)
                add_nodes_from_path(main_path, path, G)
                break


fwd_paths = sorted(fwd_paths, key=lambda p: len(p), reverse=True)
fwd_path = fwd_paths[0].copy()
fwd_path_set = set(fwd_path)
for path in fwd_paths[1:]:
    to_add = [n for n in path if n not in fwd_path_set]
    add_nodes_from_path(fwd_path, to_add, G_fwd)
    fwd_path_set = set(fwd_path)

bwd_path = fwd_path[::-1]

time_dict = nx.shortest_path_length(G, source=s, weight="time")
sidx = fwd_path.index(s)
for n, t in time_dict.items():
    if fwd_path.index(n) < sidx:
        time_dict[n] = -t

df = (
    df.lazy()
    .explode("arrname", "exp_arr_time", "exp_dep_time")
    #  .filter(pl.col("arrname").str.to_lowercase().is_in(fwd_path))
    .group_by("journey_ref")
    .agg("arrname", "exp_arr_time", "exp_dep_time")
    .collect()
)


def find_endpoint(stops: list[str], is_fwd: bool):
    last_node = stops[-1].lower()
    if is_fwd:
        for endpoint, nodes in fwd_endpoints_nodes.items():
            if last_node in nodes:
                return endpoint
        else:
            return "Unknown fwd"
    else:
        for endpoint, nodes in bwd_endpoints_nodes.items():
            if last_node in nodes:
                return endpoint
        else:
            return "Unknown bwd"


endpoints.append("Unknown fwd")
endpoints.append("Unknown bwd")
colors = {e: c for e, c in zip(endpoints, COLORS.colors)}


fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(19, 20))
for journey in df.partition_by("journey_ref"):
    times = np.array(sum(
        ([a, d] for a, d in zip(journey["exp_arr_time"][0], journey["exp_dep_time"][0])),
        start=[],
    ))
    stops = np.repeat(journey["arrname"][0], 2)
    invalid_idx = times < np.maximum.accumulate(times)
    if np.any(invalid_idx):
        times = times[~invalid_idx]
        stops = stops[~invalid_idx]
    is_fwd = fwd_path.index(stops[0].lower()) < fwd_path.index(stops[-1].lower())
    is_bwd = bwd_path.index(stops[0].lower()) < bwd_path.index(stops[-1].lower())
    #  indices = [fwd_path.index(s.lower()) for s in stops]
    indices = [time_dict[s.lower()] for s in stops]
    this_endpoint = find_endpoint(journey["arrname"][0].to_list(), is_fwd)
    color = colors[this_endpoint]
    if is_fwd:
        axs[0].plot(times, indices, "-o", alpha=0.7, markersize=1.5, linewidth=0.8, color=color)
    elif is_bwd:
        axs[1].plot(times, indices, "-o", alpha=0.7, markersize=1.2, linewidth=0.8, color=color)
    elif len(stops) > 2:
        print("Invalid journey")
axs[0].set_yticks(ticks=list(time_dict.values()), labels=list(time_dict.keys()))
axs[1].set_yticks(ticks=list(time_dict.values()), labels=list(time_dict.keys()))
ymin = min(time_dict.values())
ymax = max(time_dict.values())
axs[0].set_ylim(ymin, ymax)
axs[1].set_ylim(ymin, ymax)
xmin = df["exp_arr_time"].explode().min()
xmax = df["exp_dep_time"].explode().max()
axs[0].set_xlim(xmin, xmax)
axs[1].set_xlim(xmin, xmax)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axs[0].grid()
axs[1].grid()
axs[0].set_title(f"{FROM} -> {TO}")
axs[1].set_title(f"{TO} -> {FROM}")
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, GRAPH_NAME), dpi=300)
