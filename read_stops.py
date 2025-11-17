import os

import polars as pl
import geopandas as gpd

OUTPUT_DIR = "./data/"
OUTPUT_FILENAME = "output/all_stops.parquet"
MODES = [
    "tram",
    "metro",
    "rail",
    "monorail",
    "railway_service",
    "hsr",  # TGV
    "long_distance_rail",
    "inter_regional_rail",
    "sleeper_rail",
    "regional_rail",  # TER
    "tourist_railway",
    "rail_shuttle",
    "suburban_railway",  # RER
    "urban_railway_service",
    "metro_service",
    "underground",
    "urban_railway",
    "monorail_service",
    "tram_service",
    "city_tram",
    "local_tram",
    "regional_tram",
    "sightseeing_tram",
    "shuttle_tram",
]

all_stops = None
for directory in os.listdir(OUTPUT_DIR):
    routes = os.path.join(OUTPUT_DIR, directory, "routes.parquet")
    trips = os.path.join(OUTPUT_DIR, directory, "trips.parquet")
    sequences = os.path.join(OUTPUT_DIR, directory, "sequences.parquet")
    stops = os.path.join(OUTPUT_DIR, directory, "stops.parquet")
    if any(
        map(
            lambda f: not os.path.isfile(f) or os.stat(f).st_size == 0,
            (routes, trips, sequences, stops),
        )
    ):
        continue
    route_modes = pl.scan_parquet(routes).select(
        "route_id", mode=pl.col("route_type").cast(pl.String)
    )
    sequence_modes = (
        pl.scan_parquet(trips)
        .join(route_modes, on="route_id")
        .select("sequence_id", "mode")
        .unique()
    )
    stop_modes = (
        pl.scan_parquet(sequences)
        .join(sequence_modes, on="sequence_id")
        .select("mode", "stop_id")
        .explode("stop_id")
        .unique()
        .group_by("stop_id")
        .agg(modes="mode")
    )
    stops = (
        pl.scan_parquet(stops)
        .join(stop_modes, on="stop_id", how="left")
        .select(
            "stop_name",
            "stop_lat",
            "stop_lon",
            pl.col("location_type").cast(pl.String),
            "modes",
            slug=pl.lit(directory),
        )
        .collect()
    )
    if all_stops is None:
        all_stops = stops
    else:
        all_stops = pl.concat((all_stops, stops), how="vertical")

assert all_stops is not None
df = all_stops.unique()
print(f"Number of stops: {len(df)}")

gdf = gpd.GeoDataFrame(
    {
        "stop_name": df["stop_name"],
        "location_type": df["location_type"],
        "slug": df["slug"],
        "modes": df["modes"],
    },
    geometry=gpd.GeoSeries.from_xy(df["stop_lon"], df["stop_lat"], crs="epsg:4326"),
)
gdf.to_parquet(OUTPUT_FILENAME)
