import os
import shutil
from datetime import datetime, UTC
import json

import pytz
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

OUTPUT_FILE = os.path.join(BASE_DIR, "rera_times.parquet")
BACKUP_FILE = os.path.join(BASE_DIR, "rera_times.parquet.backup")

CLH = "STIF:StopPoint:Q:40918:"
# Both directions in the global request, with about 30min of history.
CP = "STIF:StopPoint:Q:411352:"
NANTERRE = "STIF:StopPoint:Q:40916:"
LA_DEFENSE = "STIF:StopPoint:Q:40951:"
JOINVILLE = "STIF:StopPoint:Q:40932:"
BOISSY = "STIF:StopPoint:Q:412802:"
BOISSYS = ("STIF:StopPoint:Q:412802:", "STIF:StopPoint:Q:473984:", "STIF:StopPoint:Q:473988:", "STIF:StopPoint:Q:473987:")
SUCY = ("STIF:StopPoint:Q:412803:", "STIF:StopPoint:Q:474007:", "STIF:StopPoint:Q:474010:")

RER_A = "STIF:Line::C01742:"

tz = pytz.timezone("Europe/Paris")

now = datetime.now(UTC)
print("Running request")
response = requests.get(f"{API_URL}/estimated-timetable", headers={"apiKey": API_KEY})

stop_times = list()

if response.ok:
    print("Reading response")
    data = response.json().get("Siri", dict()).get("ServiceDelivery", dict())
    response_datetime = datetime.fromisoformat(data["ResponseTimestamp"])
    timetable = data["EstimatedTimetableDelivery"][0]["EstimatedJourneyVersionFrame"][0]["EstimatedVehicleJourney"]
    rera = list(filter(lambda t: "C01742" in t.get("LineRef", dict()).get("value"), timetable))
    for trip in rera:
        longtrain = trip["VehicleFeatureRef"][0] == "longTrain"
        journey_ref = trip["DatedVehicleJourneyRef"]["value"]
        dest_ref = trip["DestinationRef"]["value"]
        dest_name = trip["DestinationName"][0]["value"]
        for call in trip["EstimatedCalls"]["EstimatedCall"]:
            exp_dep_time = call.get("ExpectedDepartureTime")
            if exp_dep_time and datetime.fromisoformat(exp_dep_time) > now:
                continue
            exp_arr_time = call.get("ExpectedArrivalTime")
            if exp_arr_time and datetime.fromisoformat(exp_arr_time) > now:
                continue
            aim_dep_time = call.get("AimedDepartureTime")
            aim_arr_time = call.get("AimedArrivalTime")
            stop_ref = call["StopPointRef"]["value"]
            dep_status = call.get("DepartureStatus")
            arr_status = call.get("ArrivalStatus")
            stop_times.append({
                "journey_ref": journey_ref,
                "dest_ref": dest_ref,
                "dest_name": dest_name,
                "stop_ref": stop_ref,
                "longtrain": longtrain,
                "dep_status": dep_status,
                "arr_status": arr_status,
                "exp_dep_time": exp_dep_time,
                "exp_arr_time": exp_arr_time,
                "aim_dep_time": aim_dep_time,
                "aim_arr_time": aim_arr_time,
            })

df = pl.DataFrame(stop_times).with_columns(
    pl.col("exp_dep_time").str.to_datetime(),
    pl.col("exp_arr_time").str.to_datetime(),
    pl.col("aim_dep_time").str.to_datetime(),
    pl.col("aim_arr_time").str.to_datetime(),
)

if os.path.isfile(OUTPUT_FILE):
    shutil.copyfile(OUTPUT_FILE, BACKUP_FILE)

    old_df = pl.scan_parquet(OUTPUT_FILE)

    df = pl.concat((df.lazy(), old_df), how="vertical", rechunk=True).unique(subset=["journey_ref", "stop_ref"], keep="first", maintain_order=True).collect()

print(len(df))
df.write_parquet(OUTPUT_FILE)
