import os
from datetime import time
import itertools

import plotly.express as px
import numpy as np
import networkx as nx
import polars as pl
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
if not os.path.isdir(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

GRAPH_NAME = "2025-09-19-A.png"

#  LINE = "STIF:Line::C01741:"  # Transilien U.
LINE = "STIF:Line::C01742:"  # RER A.
#  LINE = "STIF:Line::C01384:"  # M14.
#  LINE = "STIF:Line::C01372:"  # M2.
DATE = "2025-09-18"
#  FROM = "Cergy Le Haut"
FROM = "la défense"
#  FROM = "Gare de Poissy"
#  TO = "Boissy-Saint-Léger"
#  TO = "Gare de La Verrière"
TO = "vincennes"
COLOR = "red"

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
fwd_path_idx = {s: i for i, s in enumerate(fwd_path)}
bwd_path_idx = {s: i for i, s in enumerate(bwd_path)}

df = (
    df.lazy()
    .explode("arrname", "exp_arr_time", "exp_dep_time")
    .filter(pl.col("arrname").str.to_lowercase().is_in(fwd_path))
    .group_by("journey_ref")
    .agg("arrname", "exp_arr_time", "exp_dep_time")
    .collect()
)

df = df.with_columns(
    stop_diff=pl.col("arrname")
    .list.eval(pl.element().str.to_lowercase().replace_strict(fwd_path_idx))
    .list.diff()
    .list.max()
).with_columns(
    direction=pl.when(pl.col("stop_diff") > 0)
    .then(pl.lit("forward"))
    .when(pl.col("stop_diff") < 0)
    .then(pl.lit("backward"))
)

df = df.explode("arrname", "exp_arr_time", "exp_dep_time")
# TODO: keep case
df = df.with_columns(pl.col("arrname").str.to_lowercase())
data = pl.concat(
    (
        df.select("journey_ref", "arrname", "direction", time="exp_arr_time"),
        df.select("journey_ref", "arrname", "direction", time="exp_dep_time"),
    ),
    how="vertical",
).sort("journey_ref", "time")

fig = px.line(
    data.filter(direction="forward"),
    x="time",
    y="arrname",
    line_group="journey_ref",
    markers=True,
    hover_name="journey_ref",
)
fig.update_yaxes(categoryorder="array", categoryarray=fwd_path)
