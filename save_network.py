import os

import polars as pl
import geopandas as gpd

#  OUTPUT_DIR = "./data/reseau-urbain-et-interurbain-dile-de-france-mobilites"
#  OUTPUT_STOPS = "idf_stops.parquet"
#  OUTPUT_LINES = "idf_lines.parquet"

OUTPUT_DIR = "./data/chambery/"
OUTPUT_STOPS = "chambery_stops.parquet"
OUTPUT_LINES = "chambery_lines.parquet"

stops = (
    pl.scan_parquet(os.path.join(OUTPUT_DIR, "stops.parquet"))
    .select("stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station_id")
    .collect()
)

routes = (
    pl.scan_parquet(os.path.join(OUTPUT_DIR, "routes.parquet"))
    .select("route_id", "route_type", "route_long_name", "route_color")
    .collect()
)

route_stop_map = (
    pl.scan_parquet(os.path.join(OUTPUT_DIR, "trips.parquet"))
    .join(pl.scan_parquet(os.path.join(OUTPUT_DIR, "sequences.parquet")), on="sequence_id")
    .group_by("route_id")
    .agg(pl.col("stop_id").explode().unique())
    .explode("stop_id")
    .collect()
)

bus_stops = (
    routes.lazy()
    .filter(pl.col("route_type") == "bus")
    .join(route_stop_map.lazy(), on="route_id")
    .select("stop_id")
    .collect()
    .to_series()
)
non_bus_stops = (
    routes.lazy()
    .filter(pl.col("route_type") != "bus")
    .join(route_stop_map.lazy(), on="route_id")
    .select("stop_id")
    .collect()
    .to_series()
)

parent_children_stop_map = stops.group_by("parent_station_id").agg("stop_id")
main_stops = (
    stops.lazy()
    .filter(pl.col("parent_station_id").is_null())
    .join(parent_children_stop_map.lazy(), left_on="stop_id", right_on="parent_station_id")
    .select(
        "stop_id",
        "stop_name",
        "stop_lon",
        "stop_lat",
        "location_type",
        pl.col("stop_id_right").alias("children_stops"),
    )
    .with_columns(
        bus_stop=pl.col("children_stops").list.eval(pl.element().is_in(bus_stops)).list.any(),
        non_bus_stop=pl.col("children_stops")
        .list.eval(pl.element().is_in(non_bus_stops))
        .list.any(),
    )
    .collect()
)

stops_gdf = gpd.GeoDataFrame(
    main_stops.drop("stop_lon", "stop_lat").to_pandas(),
    geometry=gpd.GeoSeries.from_xy(main_stops["stop_lon"], main_stops["stop_lat"], crs="EPSG:4326"),
)
stops_gdf.to_parquet(OUTPUT_STOPS)
