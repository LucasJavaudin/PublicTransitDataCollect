import os
from datetime import time
import itertools

import numpy as np
import networkx as nx
import polars as pl
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
if not os.path.isdir(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

GRAPH_NAME = "2025-01-27-U.png"

LINE = "STIF:Line::C01741:"  # Transilien U.
#  LINE = "STIF:Line::C01742:"  # RER A.
#  LINE = "STIF:Line::C01384:"  # M14.
#  LINE = "STIF:Line::C01372:"  # M2.
DATE = "2025-01-28"
#  FROM = "Cergy Le Haut"
FROM = "La Défense (Grande Arche)"
#  FROM = "Gare de Poissy"
#  TO = "Boissy-Saint-Léger"
TO = "Gare de La Verrière"
COLOR = "purple"

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

stops = pl.scan_parquet(os.path.join(BASE_DIR, "arrets.parquet")).with_columns(
    stop_ref=pl.lit("STIF:StopPoint:Q:") + pl.col("arrid") + pl.lit(":")
)

df = df.join(stops, on="stop_ref")

df = df.sort("aim_arr_time", "aim_dep_time", "mean_time")

df = df.with_columns(
    is_from=(pl.col("arrname").str.to_lowercase() == FROM.lower()).cast(pl.UInt8),
    is_to=(pl.col("arrname").str.to_lowercase() == TO.lower()).cast(pl.UInt8),
)

df = df.group_by("journey_ref").agg(
    "arrname", "mean_time", "exp_arr_time", "exp_dep_time", "aim_arr_time", "aim_dep_time"
)

df = df.collect()

print(f"Number of journey: {len(df):,}")

# Find all stop pairs on the line.
direct_connections = set()
indirect_connections = set()
for journey in df.partition_by("journey_ref"):
    trip_stops = np.array(journey["arrname"][0])
    unique, idx, counts = np.unique(trip_stops, return_counts=True, return_index=True)
    idx.sort()
    uniques = trip_stops[idx]
    looped_stops = set(unique[counts > 1])
    direct_connections |= {f"{a}->{b}" for a, b in itertools.pairwise(uniques)}
    indirect_connections |= {
        f"{a}->{b}"
        for i, a in enumerate(trip_stops[:-2])
        for b in trip_stops[i + 2 :]
        if a not in looped_stops and b not in looped_stops
    }
    if "Concorde->Champs-Élysées - Clemenceau" in indirect_connections:
        break

connections = direct_connections.difference(indirect_connections)

# Find the path FROM -> TO.
G = nx.DiGraph()
for pair in connections:
    a, b = pair.split("->")
    G.add_edge(a.lower(), b.lower())

# FIXME: Required for RER A.
if LINE == "STIF:Line::C01742:":
    G.add_edge("houilles - carrières-sur-seine", "nanterre - préfecture")
    G.add_edge("nanterre - préfecture", "houilles - carrières-sur-seine")

# FIXME: Only one direction is added some times.
for a, b in G.edges():
    G.add_edge(b, a)

fwd_path = nx.dijkstra_path(G, FROM.lower(), TO.lower())
bwd_path = fwd_path[::-1]

df = (
    df.lazy()
    .explode("arrname", "exp_arr_time", "exp_dep_time")
    .filter(pl.col("arrname").str.to_lowercase().is_in(fwd_path))
    .group_by("journey_ref")
    .agg("arrname", "exp_arr_time", "exp_dep_time")
    .collect()
)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(19, 10))
for journey in df.partition_by("journey_ref"):
    times = sum(
        ([a, d] for a, d in zip(journey["exp_arr_time"][0], journey["exp_dep_time"][0])),
        start=[],
    )
    stops = np.repeat(journey["arrname"][0], 2)
    indices = [fwd_path.index(s.lower()) for s in stops]
    is_fwd = all(a <= b for a, b in itertools.pairwise(indices))
    is_bwd = all(a >= b for a, b in itertools.pairwise(indices))
    if is_fwd:
        axs[0].plot(times, indices, "-o", color=COLOR, alpha=0.7, markersize=2)
    elif is_bwd:
        axs[1].plot(times, indices, "-o", color=COLOR, alpha=0.7, markersize=2)
    else:
        print("Invalid journey")
axs[0].set_yticks(ticks=range(len(fwd_path)), labels=fwd_path)
axs[1].set_yticks(ticks=range(len(fwd_path)), labels=fwd_path)
axs[0].grid()
axs[1].grid()
axs[0].set_title(f"{FROM} -> {TO}")
axs[1].set_title(f"{TO} -> {FROM}")
fig.tight_layout()
fig.savefig(os.path.join(GRAPH_DIR, GRAPH_NAME), dpi=300)
