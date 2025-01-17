import os

import polars as pl
import geopandas as gpd

OUTPUT_DIR = "./data/"
OUTPUT_FILENAME = "all_stops.parquet"

all_stops = None
for directory in os.listdir(OUTPUT_DIR):
    filename = os.path.join(OUTPUT_DIR, directory, "stops.parquet")
    if os.path.isfile(filename):
        stops = (
            pl.scan_parquet(filename)
            .filter(pl.col("location_type").is_in(("stop", "station")))
            .select("stop_name", "stop_lat", "stop_lon", "location_type")
        )
        if all_stops is None:
            all_stops = stops
        else:
            all_stops = pl.concat((all_stops, stops), how="vertical")

assert all_stops is not None
df = all_stops.unique().collect()
print(f"Number of stops: {len(df)}")

gdf = gpd.GeoDataFrame(
    {
        "stop_name": df["stop_name"],
        "location_type": df["location_type"],
    },
    geometry=gpd.GeoSeries.from_xy(df["stop_lon"], df["stop_lat"], crs="epsg:4326"),
)
gdf.to_parquet(OUTPUT_FILENAME)
