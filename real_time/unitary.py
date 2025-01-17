import os
from datetime import datetime, UTC
import json

import requests
import polars as pl

API_URL = "https://prim.iledefrance-mobilites.fr/marketplace"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

secrets_file = os.path.join(BASE_DIR, "..", "secrets.json")
if os.path.isfile(secrets_file):
    with open(secrets_file, "r") as f:
        secrets = json.load(f)
        API_KEY = secrets["api_key"]
else:
    raise Exception(f"Cannot read API Key from `{secrets_file}`")

BOISSYS = ("STIF:StopPoint:Q:412802:", "STIF:StopPoint:Q:473984:", "STIF:StopPoint:Q:473988:", "STIF:StopPoint:Q:473987:")

RER_A = "STIF:Line::C01742:"

df = (
    pl.scan_parquet(
        "./perimetre-des-donnees-tr-disponibles-plateforme-idfm.parquet"
    )
    .filter(pl.col("line") == RER_A)
    .select("ns2_stoppointref", "ns2_stopname")
    .collect()
)

stop_times = list()

for stop_ref, stop_name in zip(df["ns2_stoppointref"], df["ns2_stopname"]):
    print(f"=== {stop_name} ===")
    now = datetime.now(UTC)
    params = {
        "MonitoringRef": stop_ref,
        "LineRef": RER_A,
    }
    response = requests.get(
        f"{API_URL}/stop-monitoring",
        headers={"apiKey": API_KEY},
        params=params,
    )
    if response.ok:
        timetable = (
            response.json()
            .get("Siri", dict())
            .get("ServiceDelivery", dict())
            .get("StopMonitoringDelivery", list())[0]
            .get("MonitoredStopVisit")
        )
        if timetable is None:
            print("Invalid response")
        for record in timetable:
            if not "MonitoredCall" in record.get("MonitoredVehicleJourney", dict()):
                print(record)
                continue
            data = record["MonitoredVehicleJourney"]
            exp_dep_time = data["MonitoredCall"].get("ExpectedDepartureTime")
            exp_arr_time = data["MonitoredCall"].get("ExpectedArrivalTime")
            past_dep_time = (
                bool(exp_dep_time)
                and datetime.fromisoformat(exp_dep_time) < now
            )
            past_arr_time = (
                bool(exp_arr_time)
                and datetime.fromisoformat(exp_arr_time) < now
            )
            if past_dep_time or (past_arr_time and exp_dep_time is None):
                aim_dep_time = data["MonitoredCall"].get("AimedDepartureTime")
                aim_arr_time = data["MonitoredCall"].get("AimedArrivalTime")
                dep_status = data["MonitoredCall"].get("DepartureStatus")
                arr_status = data["MonitoredCall"].get("ArrivalStatus")
                order = data["MonitoredCall"].get("Order")
                dest_ref = data["DestinationRef"]["value"]
                dest_name = data["DestinationName"][0]["value"]
                line = data["LineRef"]["value"]
                operator = data["OperatorRef"]["value"]
                long_train = data["VehicleFeatureRef"][0] == "longTrain"
                stop_times.append({
                    "line": line,
                    "stop_ref": stop_ref,
                    "stop_name": stop_name,
                    "operator": operator,
                    "arr_status": arr_status,
                    "dep_status": dep_status,
                    "aim_arr_time": aim_arr_time,
                    "aim_dep_time": aim_dep_time,
                    "exp_arr_time": exp_arr_time,
                    "exp_dep_time": exp_dep_time,
                    "order": order,
                    "dest_ref": dest_ref,
                    "dest_name": dest_name,
                    "long_train": long_train,
                })
    else:
        print(f"Code: {response.code}")
        print(f"Reason: {response.reason}")

stop_times_df = (
    pl.DataFrame(stop_times)
    .with_columns(
        pl.col("aim_arr_time").str.to_datetime(),
        pl.col("aim_dep_time").str.to_datetime(),
        pl.col("exp_arr_time").str.to_datetime(),
        pl.col("exp_dep_time").str.to_datetime(),
    )
)
