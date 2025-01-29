import os
from datetime import date, datetime, timezone
from zipfile import ZipFile, BadZipFile

import requests
import polars as pl
import polars.selectors as cs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "./data/")

BASE_API_URL = "https://transport.data.gouv.fr/api"

VERBOSE = False


def find_file(zipfile, filename):
    for file in zipfile.filelist:
        if file.filename.endswith(filename):
            return zipfile.open(file.filename)


def time_col_to_seconds(col_name):
    return (
        pl.col(col_name)
        .str.splitn(":", 3)
        .struct.with_fields(
            seconds=pl.field("field_0").cast(pl.UInt32) * 3600
            + pl.field("field_1").cast(pl.UInt32) * 60
            + pl.field("field_2").cast(pl.UInt32),
        )
        .struct.field("seconds")
        .alias(col_name)
    )


def read_update(slug):
    output_dir = os.path.join(OUTPUT_DIR, slug)
    filename = os.path.join(output_dir, "last_update.txt")
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            last_date = datetime.fromisoformat(f.read())
        if last_date.tzinfo is None:
            last_date = last_date.replace(tzinfo=timezone.utc)
        return last_date
    else:
        return datetime(1, 1, 1, tzinfo=timezone.utc)


def dataset_needs_update(dataset):
    if dataset.get("type", "") != "public-transit":
        return False
    if dataset["updated"] is None:
        return False
    slug = dataset["slug"]
    last_collect_update = read_update(slug)
    last_server_update = datetime.fromisoformat(dataset["updated"])
    return last_server_update > last_collect_update


def request_gtfs_files():
    print(datetime.now())
    response = requests.get(f"{BASE_API_URL}/datasets")
    if response.ok:
        datasets = list(filter(lambda it: dataset_needs_update(it), response.json()))
        n = len(datasets)
        for i, dataset in enumerate(datasets):
            try:
                slug = dataset["slug"]
                print(f"\n=== Dataset ({i + 1}/{n}) {slug} ===\n")
                update_dataset(dataset)
            except Exception as e:
                print("Error. Failed to read dataset.")
                print(e)
    else:
        print("Error retrieving GTFS files with API")
        print(f"Code: {response.status_code}")
        print(f"Reason: {response.reason}")


def update_dataset(dataset):
    slug = dataset["slug"]
    response = requests.get(f"{BASE_API_URL}/datasets/{dataset['id']}")
    if not response.ok:
        print(f"Warning. Failed to retrieve API data for dataset {slug}")
        return
    resources = list(
        filter(lambda r: r["payload"]["format"] == "GTFS", response.json().get("history", []))
    )
    if not resources:
        print(f"Warning. No GTFS file to read for dataset {slug}")
        return
    output_dir = os.path.join(OUTPUT_DIR, slug)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    last_update = read_update(slug)
    resources = list(
        sorted(
            filter(lambda r: datetime.fromisoformat(r["updated_at"]) > last_update, resources),
            key=lambda r: r["updated_at"],
        )
    )
    if resources:
        read_and_load_history(resources, output_dir)
    else:
        print(f"Already up to date for dataset {slug}")
    # Refresh last update.
    try:
        filename = os.path.join(output_dir, "last_update.txt")
        update = datetime.fromisoformat(dataset["updated"])
        with open(filename, "w") as f:
            f.write(str(update))
    except Exception as e:
        print("Warning. Failed to set last update!")
        print(e)


def read_and_load_history(resources, output_dir):
    n = len(resources)
    for i, resource in enumerate(resources):
        try:
            url = resource["payload"]["permanent_url"]
            modified_date = datetime.fromisoformat(resource["updated_at"]).date()
            print(f"Downloading ({i + 1}/{n}) from {url}")
            download_path = os.path.join(BASE_DIR, "tmp", "gtfs.zip")
            download_zip(url, download_path)
            read_and_merge(download_path, output_dir, modified_date)
            os.remove(download_path)
        except Exception as e:
            print("Warning. Failed to read resource!")
            print(e)


def read_and_load_csv_history(history_filename, output_dir):
    history = pl.read_csv(history_filename).sort("resource_history_id")
    n = len(history)
    for i, url in enumerate(history["permanent_url"]):
        print(f"Downloading ({i + 1}/{n}) from {url}")
        download_path = os.path.join(BASE_DIR, "tmp", "gtfs.zip")
        download_zip(url, download_path)
        read_and_merge(download_path, output_dir)
        os.remove(download_path)


def download_zip(url: str, output_filename: str):
    if not os.path.isdir(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    response = requests.get(url, stream=True)
    with open(output_filename, "wb") as file:
        file.write(response.content)


def read_and_merge(input_zipfilename, output_dir, modified_date):
    try:
        input_zipfile = ZipFile(input_zipfilename)
    except BadZipFile:
        print("Warning. Skipping invalid zipfile")
        return

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    ############
    #  agency  #
    ############

    if VERBOSE:
        print("Collecting agencies")
    zipped_agencies = find_file(input_zipfile, "agency.txt")
    if zipped_agencies is None:
        raise Exception("Missing file: `agency.txt`")
    agencies = pl.scan_csv(zipped_agencies.read(), schema_overrides={"agency_name": pl.String})
    available_columns = agencies.collect_schema().names()
    columns = ["agency_name"]
    if "agency_id" in available_columns:
        columns.append(pl.col("agency_id").cast(pl.String).alias("original_agency_id"))
    else:
        columns.append(pl.lit("default", dtype=pl.String).alias("original_agency_id"))
    agencies = agencies.select(columns)
    filename = os.path.join(output_dir, "agency.parquet")
    if os.path.isfile(filename):
        previous_agencies = pl.scan_parquet(filename).drop("agency_id")
        agencies = pl.concat((previous_agencies, agencies), how="vertical", rechunk=True).unique(
            keep="first", maintain_order=True
        )
    agencies = agencies.with_columns(agency_id=pl.int_range(pl.len(), dtype=pl.UInt32))
    agencies = agencies.collect()
    agency_id_map = agencies.select("original_agency_id", "agency_id").unique(
        subset="original_agency_id", keep="last"
    )

    ############
    #  routes  #
    ############

    if VERBOSE:
        print("Collecting routes")
    zipped_routes = find_file(input_zipfile, "routes.txt")
    if zipped_routes is None:
        raise Exception("Missing file: `routes.txt`")
    routes = pl.scan_csv(
        zipped_routes.read(),
        schema_overrides={
            "route_id": pl.String,
            "route_type": pl.UInt16,
            "route_short_name": pl.String,
            "route_sort_order": pl.UInt32,
            "route_color": pl.String,
            "route_text_color": pl.String,
        },
    )
    available_columns = routes.collect_schema().names()
    # https://developers.google.com/transit/gtfs/reference/extended-route-types
    route_types = {
        0: "tram",
        1: "metro",
        2: "rail",
        3: "bus",
        4: "ferry",
        5: "cable_tram",
        6: "aerial_lift",
        7: "funicular",
        11: "trolleybus",
        12: "monorail",
        100: "railway_service",
        101: "hsr",  # TGV
        102: "long_distance_rail",
        103: "inter_regional_rail",
        105: "sleeper_rail",
        106: "regional_rail",  # TER
        107: "tourist_railway",
        108: "rail_shuttle",
        109: "suburban_railway",  # RER
        200: "coach_service",
        201: "international_coach",
        202: "national_coach",
        203: "shuttle_coach",
        204: "regional_coach",
        400: "urban_railway_service",
        401: "metro_service",
        402: "underground",
        403: "urban_railway",
        405: "monorail_service",
        700: "bus_service",
        701: "regional_bus",
        702: "express_bus",
        703: "stopping_bus",
        704: "local_bus",
        705: "night_bus",
        706: "post_bus",
        712: "school_bus",
        715: "demand_and_response_bus",
        800: "trolleybus_service",
        900: "tram_service",
        901: "city_tram",
        902: "local_tram",
        903: "regional_tram",
        904: "sightseeing_tram",
        905: "shuttle_tram",
        1000: "water_transport_service",
        1100: "air_service",
        1200: "ferry_service",
        1300: "aerial_lift_service",
        1301: "telecabin",
        1400: "funicular_service",
        1500: "taxi_service",
        1501: "communal_service",
        1700: "miscellaneous_service",
        1702: "horse-drawn_carriage",
    }
    enum = pl.Enum(route_types.values())
    columns = [
        pl.col("route_id").alias("original_route_id"),
        pl.col("route_type").replace_strict(route_types, default=None).cast(enum),
    ]
    if "agency_id" in available_columns:
        columns.append(pl.col("agency_id").cast(pl.String).alias("original_agency_id"))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("original_agency_id"))
    if "route_short_name" in available_columns:
        columns.append(pl.col("route_short_name").cast(pl.String))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("route_short_name"))
    if "route_long_name" in available_columns:
        columns.append(pl.col("route_long_name").cast(pl.String))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("route_long_name"))
    if "route_color" in available_columns:
        columns.append(pl.col("route_color").cast(pl.String))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("route_color"))
    if "route_text_color" in available_columns:
        columns.append(pl.col("route_text_color").cast(pl.String))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("route_text_color"))
    if "route_sort_order" in available_columns:
        columns.append(pl.col("route_sort_order").cast(pl.UInt32, strict=False))
    else:
        columns.append(pl.lit(None, dtype=pl.UInt32).alias("route_sort_order"))
    if "network_id" in available_columns:
        columns.append("network_id")
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("network_id"))
    routes = routes.select(columns)
    routes = routes.with_columns(
        agency_id=pl.col("original_agency_id").replace_strict(
            agency_id_map["original_agency_id"], agency_id_map["agency_id"]
        )
    )
    filename = os.path.join(output_dir, "routes.parquet")
    if os.path.isfile(filename):
        previous_routes = pl.scan_parquet(filename).drop("route_id")
        routes = pl.concat((previous_routes, routes), how="vertical", rechunk=True).unique(
            keep="first", maintain_order=True
        )
    routes = routes.with_columns(route_id=pl.int_range(pl.len(), dtype=pl.UInt32))
    routes = routes.collect()
    route_id_map = routes.select("original_route_id", "route_id").unique(
        subset="original_route_id", keep="last"
    )

    ###########
    #  stops  #
    ###########

    if VERBOSE:
        print("Collecting stops")
    zipped_stops = find_file(input_zipfile, "stops.txt")
    if zipped_stops is None:
        raise Exception("Missing file: `stops.txt`")
    stops = pl.scan_csv(
        zipped_stops.read(),
        schema_overrides={
            "stop_id": pl.String,
            "stop_name": pl.String,
            "stop_lat": pl.Float64,
            "stop_lon": pl.Float64,
            "location_type": pl.UInt8,
        },
    )
    available_columns = stops.collect_schema().names()
    columns = [pl.col("stop_id").alias("original_stop_id"), "stop_name", "stop_lat", "stop_lon"]
    location_types = {
        0: "stop",
        1: "station",
        2: "entrance/exit",
        3: "generic_node",
        4: "boarding_area",
    }
    enum = pl.Enum(location_types.values())
    if "location_type" in available_columns:
        columns.append(
            pl.col("location_type")
            .cast(pl.UInt8)
            .fill_null(0)
            .replace_strict(location_types, default=None)
            .cast(enum)
        )
    else:
        columns.append(pl.lit("stop").cast(enum).alias("location_type"))
    if "parent_station" in available_columns:
        columns.append(pl.col("parent_station").cast(pl.String).alias("original_parent_station_id"))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("original_parent_station_id"))
    stops = stops.select(columns).collect()
    filename = os.path.join(output_dir, "stops.parquet")
    if os.path.isfile(filename):
        previous_stops = pl.read_parquet(filename)
        # Add new stops and stops with updated characteristics.
        # Columns `stop_id` and `parent_station_id` are null at that point for the newly added stops.
        all_stops = pl.concat((previous_stops, stops), how="diagonal", rechunk=True).unique(
            subset=cs.exclude("stop_id", "parent_station_id"), maintain_order=True
        )
        # Add stops that were not updated but whose parent station was updated (we do it twice to handle
        # parents' of parents).
        for _ in range(2):
            added_stops = all_stops.filter(pl.col("stop_id").is_null())["original_stop_id"]
            to_add_stops = stops.filter(
                pl.col("original_stop_id").is_in(added_stops).not_(),
                pl.col("original_parent_station_id").is_in(added_stops),
            )
            all_stops = pl.concat((all_stops, to_add_stops), how="diagonal", rechunk=True)
    else:
        all_stops = stops.with_columns(parent_station_id=pl.lit(None))
    all_stops = all_stops.with_columns(stop_id=pl.int_range(pl.len(), dtype=pl.UInt32))
    stop_id_map = all_stops.select("original_stop_id", "stop_id").unique(
        subset="original_stop_id", keep="last"
    )
    all_stops = all_stops.with_columns(
        parent_station_id=pl.when(pl.col("parent_station_id").is_null())
        .then(
            pl.col("original_parent_station_id").replace_strict(
                stop_id_map["original_stop_id"], stop_id_map["stop_id"], default=None
            )
        )
        .otherwise("parent_station_id")
    )

    ################
    #  Stop times  #
    ################

    if VERBOSE:
        print("Collecting stop_times")
    zipped_stop_times = find_file(input_zipfile, "stop_times.txt")
    if zipped_stop_times is None:
        raise Exception("Missing file: `stop_times.txt`")
    stop_times = (
        pl.scan_csv(
            zipped_stop_times.read(),
            schema_overrides={
                "trip_id": pl.String,
                "arrival_time": pl.String,
                "departure_time": pl.String,
                "stop_id": pl.String,
                "stop_sequence": pl.UInt16,
            },
        )
        .sort("trip_id", "stop_sequence")
        .with_columns(time_col_to_seconds("arrival_time"), time_col_to_seconds("departure_time"))
        .with_columns(
            stopping_time=pl.col("departure_time") - pl.col("arrival_time"),
            between_stop_time=pl.col("arrival_time").shift(-1).over("trip_id")
            - pl.col("departure_time"),
        )
        .with_columns(
            pl.col("stop_id").replace_strict(
                stop_id_map["original_stop_id"], stop_id_map["stop_id"]
            )
        )
    )
    available_columns = stop_times.collect_schema().names()
    columns: list = [
        "trip_id",
        "arrival_time",
        "stopping_time",
        "between_stop_time",
        "stop_id",
    ]
    types = {
        0: "allowed",
        1: "forbidden",
        2: "must_phone",
        3: "must_coordinate",
    }
    enum = pl.Enum(types.values())
    if "pickup_type" in available_columns:
        columns.append(
            pl.col("pickup_type").cast(pl.UInt8).replace_strict(types, default=None).cast(enum)
        )
    else:
        columns.append(pl.lit(None, dtype=enum).alias("pickup_type"))
    if "drop_off_type" in available_columns:
        columns.append(
            pl.col("drop_off_type").cast(pl.UInt8).replace_strict(types, default=None).cast(enum)
        )
    else:
        columns.append(pl.lit(None, dtype=enum).alias("drop_off_type"))
    if "timepoint" in available_columns:
        columns.append(pl.col("timepoint").cast(pl.Boolean, strict=False).alias("exact_times"))
    else:
        columns.append(pl.lit(None, dtype=pl.Boolean).alias("exact_times"))
    stop_times = stop_times.select(columns)
    stop_times = stop_times.collect()

    if VERBOSE:
        print("Creating stop sequences")
    sequences = (
        stop_times.lazy()
        .group_by("trip_id", maintain_order=True)
        .agg("stop_id", "pickup_type", "drop_off_type")
        .drop("trip_id")
        .unique(maintain_order=True)
    )

    filename = os.path.join(output_dir, "sequences.parquet")
    if os.path.isfile(filename):
        previous_sequences = pl.scan_parquet(filename).drop("sequence_id")
        sequences = pl.concat((previous_sequences, sequences), how="vertical", rechunk=True).unique(
            keep="first", maintain_order=True
        )
    sequences = sequences.with_columns(sequence_id=pl.int_range(pl.len(), dtype=pl.UInt32))
    sequences = sequences.collect()

    if VERBOSE:
        print("Creating stop timings")
    timings = (
        stop_times.lazy()
        .group_by("trip_id", maintain_order=True)
        .agg(
            "stopping_time",
            "between_stop_time",
            "stop_id",
            "pickup_type",
            "drop_off_type",
        )
        .drop("trip_id")
        .unique(maintain_order=True)
        .join(sequences.lazy(), on=["stop_id", "pickup_type", "drop_off_type"], how="left")
        .select("stopping_time", "between_stop_time", "sequence_id")
    )

    filename = os.path.join(output_dir, "timings.parquet")
    if os.path.isfile(filename):
        previous_timings = pl.scan_parquet(filename).drop("timing_id")
        timings = pl.concat((previous_timings, timings), how="vertical", rechunk=True).unique(
            keep="first", maintain_order=True
        )
    timings = timings.with_columns(timing_id=pl.int_range(pl.len(), dtype=pl.UInt32))
    timings = timings.collect()

    if VERBOSE:
        print("Creating trip sequence and timings")
    trip_stop_times = (
        stop_times.lazy()
        .group_by("trip_id", maintain_order=True)
        .agg(
            "stopping_time",
            "between_stop_time",
            "stop_id",
            "pickup_type",
            "drop_off_type",
            start_time=pl.col("arrival_time").first(),
        )
        .join(sequences.lazy(), on=["stop_id", "pickup_type", "drop_off_type"], how="left")
        .join(timings.lazy(), on=["stopping_time", "between_stop_time", "sequence_id"], how="left")
        .select("trip_id", "start_time", "timing_id", "sequence_id")
        .collect()
    )

    ###########
    #  Trips  #
    ###########

    if VERBOSE:
        print("Collecting trips")
    zipped_trips = find_file(input_zipfile, "trips.txt")
    if zipped_trips is None:
        raise Exception("Missing file: `trips.txt`")
    trips = (
        pl.scan_csv(
            zipped_trips.read(),
            schema_overrides={
                "route_id": pl.String,
                "service_id": pl.String,
                "trip_id": pl.String,
                "trip_headsign": pl.String,
                "trip_short_name": pl.String,
            },
        )
        .with_columns(
            route_id=pl.col("route_id").replace_strict(
                route_id_map["original_route_id"], route_id_map["route_id"]
            )
        )
        .join(trip_stop_times.lazy(), on="trip_id", how="left")
        .rename({"trip_id": "original_trip_id"})
    )
    available_columns = trips.collect_schema().names()
    columns = ["route_id", "original_trip_id", "start_time", "timing_id", "sequence_id"]
    if "trip_headsign" in available_columns:
        columns.append(pl.col("trip_headsign").cast(pl.String))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("trip_headsign"))
    if "trip_short_name" in available_columns:
        columns.append(pl.col("trip_short_name").cast(pl.String))
    else:
        columns.append(pl.lit(None, dtype=pl.String).alias("trip_short_name"))
    if "direction_id" in available_columns:
        columns.append(
            pl.col("direction_id")
            .cast(pl.UInt8)
            .cast(pl.Boolean, strict=False)
            .alias("opposite_direction")
        )
    else:
        columns.append(pl.lit(None, dtype=pl.Boolean).alias("opposite_direction"))
    bikes_allowed = {
        0: "unknown",
        1: "yes",
        2: "no",
    }
    enum = pl.Enum(bikes_allowed.values())
    if "bikes_allowed" in available_columns:
        columns.append(
            pl.col("bikes_allowed")
            .cast(pl.UInt8)
            .fill_null(0)
            .replace_strict(bikes_allowed, default=None)
            .cast(enum)
        )
    else:
        columns.append(pl.lit(None, dtype=enum).alias("bikes_allowed"))
    trips = trips.select(columns + ["service_id"]).collect()
    original_trips = trips.select("original_trip_id", "service_id")
    trips = trips.drop("service_id")

    filename = os.path.join(output_dir, "trips.parquet")
    if os.path.isfile(filename):
        previous_trips = pl.read_parquet(filename).drop("trip_id")
        all_trips = pl.concat(
            (previous_trips, trips.drop("original_trip_id")), how="vertical", rechunk=True
        ).unique(keep="first", maintain_order=True)
    else:
        all_trips = trips.drop("original_trip_id").unique(maintain_order=True)
    join_columns = all_trips.columns
    all_trips = all_trips.with_columns(trip_id=pl.int_range(pl.len(), dtype=pl.UInt32))
    trip_id_map = trips.join(all_trips, on=join_columns, how="left", join_nulls=True).select(
        "original_trip_id", "trip_id"
    )

    ###############
    #  Transfers  #
    ###############

    if VERBOSE:
        print("Collecting transfers")
    transfers = None
    zipped_transfers = find_file(input_zipfile, "transfers.txt")
    if zipped_transfers is not None:
        transfers = pl.scan_csv(
            zipped_transfers.read(),
            schema_overrides={
                "from_stop_id": pl.String,
                "to_stop_id": pl.String,
                "from_route_id": pl.String,
                "to_route_id": pl.String,
                "from_trip_id": pl.String,
                "to_trip_id": pl.String,
                "transfer_type": pl.UInt8,
                "min_transfer_time": pl.UInt32,
            },
        )
        available_columns = transfers.collect_schema().names()
        transfer_types = {
            0: "recommended_transfer",
            1: "timed_transfer",
            2: "minimum_time",
            3: "unfeasible_transfer",
            4: "sequential_trips_in-seat_transfer",
            5: "sequential_trips_alight_transfer",
        }
        enum = pl.Enum(transfer_types.values())
        columns = [
            pl.col("from_stop_id").replace_strict(
                stop_id_map["original_stop_id"], stop_id_map["stop_id"]
            ),
            pl.col("to_stop_id").replace_strict(
                stop_id_map["original_stop_id"], stop_id_map["stop_id"]
            ),
            pl.col("transfer_type").replace_strict(transfer_types, default=None).cast(enum),
        ]
        if "from_route_id" in available_columns:
            columns.append(
                pl.col("from_route_id").replace_strict(
                    route_id_map["original_route_id"], route_id_map["route_id"], default=None
                )
            )
        else:
            columns.append(pl.lit(None, dtype=pl.UInt32).alias("from_route_id"))
        if "to_route_id" in available_columns:
            columns.append(
                pl.col("to_route_id").replace_strict(
                    route_id_map["original_route_id"], route_id_map["route_id"], default=None
                )
            )
        else:
            columns.append(pl.lit(None, dtype=pl.UInt32).alias("to_route_id"))
        if "from_trip_id" in available_columns:
            columns.append(
                pl.col("from_trip_id").replace_strict(
                    trip_id_map["original_trip_id"], trip_id_map["trip_id"], default=None
                )
            )
        else:
            columns.append(pl.lit(None, dtype=pl.UInt32).alias("from_trip_id"))
        if "to_trip_id" in available_columns:
            columns.append(
                pl.col("to_trip_id").replace_strict(
                    trip_id_map["original_trip_id"], trip_id_map["trip_id"], default=None
                )
            )
        else:
            columns.append(pl.lit(None, dtype=pl.UInt32).alias("to_trip_id"))
        if "min_transfer_time" in available_columns:
            columns.append("min_transfer_time")
        else:
            columns.append(pl.lit(None, dtype=pl.UInt32).alias("min_transfer_time"))
        transfers = transfers.select(columns)
        filename = os.path.join(output_dir, "transfers.parquet")
        if os.path.isfile(filename):
            previous_transfers = pl.scan_parquet(filename)
            transfers = pl.concat(
                (previous_transfers, transfers), how="vertical", rechunk=True
            ).unique(keep="first", maintain_order=True)
        transfers = transfers.collect()

    ##############
    #  Calendar  #
    ##############

    if VERBOSE:
        print("Processing calendars")
    start_date = date(9999, 1, 1)
    end_date = date(1, 1, 1)
    zipped_calendar = find_file(input_zipfile, "calendar.txt")
    if zipped_calendar is not None:
        calendar = pl.read_csv(
            zipped_calendar.read(),
            schema_overrides={
                "service_id": pl.String,
                "monday": pl.UInt8,
                "tuesday": pl.UInt8,
                "wednesday": pl.UInt8,
                "thursday": pl.UInt8,
                "friday": pl.UInt8,
                "saturday": pl.UInt8,
                "sunday": pl.UInt8,
                "start_date": pl.String,
                "end_date": pl.String,
            },
        ).with_columns(
            pl.col("start_date").str.strip_chars().str.to_date("%Y%m%d"),
            pl.col("end_date").str.strip_chars().str.to_date("%Y%m%d"),
        )
        if not calendar.is_empty():
            start_date = min(start_date, calendar.select(pl.col("start_date").min()).item())
            end_date = max(end_date, calendar.select(pl.col("end_date").max()).item())
    else:
        calendar = None
    zipped_calendar_dates = find_file(input_zipfile, "calendar_dates.txt")
    if zipped_calendar_dates is not None:
        calendar_dates = pl.read_csv(
            zipped_calendar_dates.read(),
            schema_overrides={
                "service_id": pl.String,
                "date": pl.String,
                "exception_type": pl.UInt8,
            },
        ).with_columns(pl.col("date").str.strip_chars().str.to_date("%Y%m%d"))
        if not calendar_dates.is_empty():
            start_date = min(start_date, calendar_dates.select(pl.col("date").min()).item())
            end_date = max(end_date, calendar_dates.select(pl.col("date").max()).item())
    else:
        calendar_dates = None
    assert (
        start_date.year != 9999 and end_date.year != 1
    ), "Either calendar.txt or calendar_dates.txt must be provided"

    # Start date cannot be prior to the GTFS file modification date.
    start_date = max(start_date, modified_date)

    if VERBOSE:
        print("Finding trips by date")
    WEEKDAYS = {
        0: "monday",
        1: "tuesday",
        2: "wednesday",
        3: "thursday",
        4: "friday",
        5: "saturday",
        6: "sunday",
    }
    trips_by_day = dict()
    for d in pl.date_range(start_date, end_date, eager=True):
        weekday = WEEKDAYS[d.weekday()]
        active_services = set()
        if calendar is not None:
            active_services |= set(
                calendar.filter(
                    pl.col("start_date") <= d, pl.col("end_date") >= d, pl.col(weekday) == 1
                )["service_id"]
            )
        if calendar_dates is not None:
            active_services |= set(
                calendar_dates.filter(pl.col("date") == d, pl.col("exception_type") == 1)[
                    "service_id"
                ]
            )
            active_services -= set(
                calendar_dates.filter(pl.col("date") == d, pl.col("exception_type") == 2)[
                    "service_id"
                ]
            )
        active_trips = original_trips.filter(pl.col("service_id").is_in(active_services))[
            "original_trip_id"
        ]
        trips_by_day[d] = active_trips

    trip_dates = pl.DataFrame(
        {
            "date": trips_by_day.keys(),
            "original_trip_id": trips_by_day.values(),
        }
    ).select(
        "date",
        trip_id=pl.col("original_trip_id").list.eval(
            pl.element().replace_strict(trip_id_map["original_trip_id"], trip_id_map["trip_id"])
        ),
    )

    filename = os.path.join(output_dir, "trip_dates.parquet")
    if os.path.isfile(filename):
        # Read the previous version of the file and exclude the day that were updated.
        previous_trip_dates = (
            pl.scan_parquet(filename).filter(pl.col("date") < start_date).collect()
        )
        trip_dates = pl.concat((previous_trip_dates, trip_dates), how="vertical", rechunk=True)

    if VERBOSE:
        print("Saving output")
    agencies.write_parquet(os.path.join(output_dir, "agencies.parquet"))
    routes.write_parquet(os.path.join(output_dir, "routes.parquet"))
    all_stops.write_parquet(os.path.join(output_dir, "stops.parquet"))
    sequences.write_parquet(os.path.join(output_dir, "sequences.parquet"))
    timings.write_parquet(os.path.join(output_dir, "timings.parquet"))
    all_trips.write_parquet(os.path.join(output_dir, "trips.parquet"))
    if transfers is not None:
        transfers.write_parquet(os.path.join(output_dir, "transfers.parquet"))
    trip_dates.write_parquet(os.path.join(output_dir, "trip_dates.parquet"))
    if VERBOSE:
        print("Done")


if __name__ == "__main__":
    #  read_and_load_csv_history(
    #  "./data/idf/historisation-dataset-668-2025-01-10.csv", "./data/idf/output"
    #  )
    #  read_and_merge("./data/idf/IDFM-gtfs.zip", "./data/idf/output/")
    request_gtfs_files()
