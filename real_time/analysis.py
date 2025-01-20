import polars as pl

df = pl.read_parquet("./rera_times.parquet")

arrets = pl.scan_parquet("./arrets.parquet")
arrets = arrets.with_columns(
    stop_ref=pl.lit("STIF:StopPoint:Q:") + pl.col("arrid") + pl.lit(":")
)

zones = pl.scan_parquet("./zones.parquet")

stop_names = arrets.join(zones, on="zdaid", how="left").select("stop_ref", "zdaname").collect()

assert df["stop_ref"].is_in(stop_names["stop_ref"]).all(), "Missing stop"

df = df.join(stop_names, on="stop_ref")

print("Stop frequency:")
print(df["zdaname"].value_counts(sort=True))
