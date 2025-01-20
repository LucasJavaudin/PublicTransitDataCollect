import os
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

OUTPUT_DIR = os.path.join(BASE_DIR, "data")
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

lines = {
    "STIF:Line::C01372:",
    "STIF:Line::C01374:",
    "STIF:Line::C01378:",
    "STIF:Line::C01381:",
    "STIF:Line::C01383:",
    "STIF:Line::C01391:",
    "STIF:Line::C01738:",
    "STIF:Line::C01744:",
    "STIF:Line::C01747:",
    "STIF:Line::C02375:",
    "STIF:Line::C01371:",
    "STIF:Line::C01382:",
    "STIF:Line::C01386:",
    "STIF:Line::C01679:",
    "STIF:Line::C01746:",
    "STIF:Line::C01795:",
    "STIF:Line::C01843:",
    "STIF:Line::C01857:",
    "STIF:Line::C02368:",
    "STIF:Line::C02370:",
    "STIF:Line::C02711:",
    "STIF:Line::C00563:",
    "STIF:Line::C01375:",
    "STIF:Line::C01390:",
    "STIF:Line::C01728:",
    "STIF:Line::C01729:",
    "STIF:Line::C01731:",
    "STIF:Line::C01736:",
    "STIF:Line::C01737:",
    "STIF:Line::C01740:",
    "STIF:Line::C01741:",
    "STIF:Line::C01742:",
    "STIF:Line::C01743:",
    "STIF:Line::C01745:",
    "STIF:Line::C01774:",
    "STIF:Line::C01794:",
    "STIF:Line::C01999:",
    "STIF:Line::C01376:",
    "STIF:Line::C01380:",
    "STIF:Line::C01387:",
    "STIF:Line::C01388:",
    "STIF:Line::C01389:",
    "STIF:Line::C01684:",
    "STIF:Line::C01739:",
    "STIF:Line::C02317:",
    "STIF:Line::C02528:",
    "STIF:Line::C02732:",
    "STIF:Line::C01373:",
    "STIF:Line::C01377:",
    "STIF:Line::C01379:",
    "STIF:Line::C01384:",
    "STIF:Line::C01727:",
    "STIF:Line::C01730:",
    "STIF:Line::C01748:",
    "STIF:Line::C01863:",
    "STIF:Line::C02344:",
    "STIF:Line::C02372:",
    "STIF:Line::C02529:",
}

tz = pytz.timezone("Europe/Paris")

now = datetime.now(UTC)
print("Running request")
response = requests.get(f"{API_URL}/estimated-timetable", headers={"apiKey": API_KEY})

stop_times = list()

if response.ok:
    print("Reading response")
    data = response.json().get("Siri", dict()).get("ServiceDelivery", dict())
    response_datetime = datetime.fromisoformat(data["ResponseTimestamp"])
    timetable = data["EstimatedTimetableDelivery"][0]["EstimatedJourneyVersionFrame"][0][
        "EstimatedVehicleJourney"
    ]
    timetable = list(filter(lambda t: t.get("LineRef", dict()).get("value") in lines, timetable))
    for trip in timetable:
        try:
            longtrain = trip["VehicleFeatureRef"][0] == "longTrain"
        except IndexError:
            longtrain = None
        line_ref = trip["LineRef"]["value"]
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
            stop_times.append(
                {
                    "line_ref": line_ref,
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
                }
            )

df = pl.DataFrame(stop_times).with_columns(
    pl.col("exp_dep_time").str.to_datetime(),
    pl.col("exp_arr_time").str.to_datetime(),
    pl.col("aim_dep_time").str.to_datetime(),
    pl.col("aim_arr_time").str.to_datetime(),
)

output_filename = os.path.join(OUTPUT_DIR, f"{now.year}-{now.month:02}-{now.day:02}.parquet")

if os.path.isfile(output_filename):
    old_df = pl.scan_parquet(output_filename)

    df = (
        pl.concat((df.lazy(), old_df), how="vertical", rechunk=True)
        .unique(subset=["journey_ref", "stop_ref"], keep="first", maintain_order=True)
        .collect()
    )

print(len(df))
df.write_parquet(output_filename)
