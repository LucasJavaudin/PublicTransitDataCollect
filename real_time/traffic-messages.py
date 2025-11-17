import os
from datetime import datetime
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

lines = [
    "C01372",
    "C01374",
    "C01378",
    "C01381",
    "C01383",
    "C01391",
    "C01738",
    "C01744",
    "C01747",
    "C02375",
    "C01371",
    "C01382",
    "C01386",
    "C01679",
    "C01746",
    "C01795",
    "C01843",
    "C01857",
    "C02368",
    "C02370",
    "C02711",
    "C00563",
    "C01375",
    "C01390",
    "C01728",
    "C01729",
    "C01731",
    "C01736",
    "C01737",
    "C01740",
    "C01741",
    "C01742",
    "C01743",
    "C01745",
    "C01774",
    "C01794",
    "C01999",
    "C01376",
    "C01380",
    "C01387",
    "C01388",
    "C01389",
    "C01684",
    "C01739",
    "C02317",
    "C02528",
    "C02732",
    "C01373",
    "C01377",
    "C01379",
    "C01384",
    "C01727",
    "C01730",
    "C01748",
    "C01863",
    "C02344",
    "C02372",
    "C02529",
]


def is_valid_disruption(disruption: dict):
    return (
        (disruption["status"] == "active")
        and ("Ascenseur" not in disruption.get("tags", []))
        and (disruption.get("category") != "Communication")
    )


def get_from_to_stops(impacted_object: dict):
    inner = None
    if "impacted_rail_section" in impacted_object:
        inner = impacted_object["impacted_rail_section"]
    if "impacted_section" in impacted_object:
        inner = impacted_object["impacted_section"]
    if inner is None:
        return
    if "from" not in inner or "to" not in inner:
        return None
    from_stop = inner["from"]
    to_stop = inner["to"]
    return {
        "from": {
            "id": from_stop["id"],
            "name": from_stop["name"],
        },
        "to": {
            "id": to_stop["id"],
            "name": to_stop["name"],
        },
    }


def is_valid_period(period: dict, now: datetime):
    begin = datetime.fromisoformat(period["begin"]).astimezone(tz)
    end = datetime.fromisoformat(period["end"]).astimezone(tz)
    return begin <= now and end >= now

tz = pytz.timezone("Europe/Paris")

now = datetime.now(tz)
now_str = now.strftime("%Y%m%dT%H%M%S")
params = {
    "disable_geojson": True,
    #  "until": now_str,
}

disruptions = list()


for line in lines:
    print(line)
    response = requests.get(
        f"{API_URL}/v2/navitia/line_reports/lines/line:IDFM:{line}/line_reports",
        headers={"apiKey": API_KEY},
        params=params,
    )
    if response.ok:
        data = response.json()
        for disruption in filter(is_valid_disruption, data["disruptions"]):
            title = next(
                map(
                    lambda m: m["text"],
                    filter(lambda m: m["channel"]["name"] == "titre", disruption["messages"]),
                )
            )
            message = next(
                map(
                    lambda m: m["text"],
                    filter(lambda m: m["channel"]["name"] == "moteur", disruption["messages"]),
                )
            )
            impacted_objects = filter(
                lambda o: o["pt_object"]["id"] == f"line:IDFM:{line}",
                disruption["impacted_objects"],
            )
            from_to_stops = list(
                filter(lambda x: x is not None, map(get_from_to_stops, impacted_objects))
            )
            period = next(
                filter(lambda p: is_valid_period(p, now), disruption["application_periods"])
            )
            begin = datetime.fromisoformat(period["begin"]).astimezone(tz)
            x = {
                "id": disruption["disruption_id"],
                "line": line,
                "start": begin,
                "end": now,
                "cause": disruption["cause"],
                "category": disruption["category"],
                "severity": disruption["severity"],
                "tags": disruption.get("tags", []),
                "title": title,
                "message": message,
                "from_to": from_to_stops,
            }
            disruptions.append(x)


stop_struct = pl.Struct({"id": pl.String, "name": pl.String})
schema = {
    "id": pl.String,
    "line": pl.String,
    "start": pl.Datetime,
    "end": pl.Datetime,
    "cause": pl.String,
    "category": pl.String,
    "severity": pl.Struct(
        {"name": pl.String, "effect": pl.String, "color": pl.String, "priority": pl.Int64}
    ),
    "tags": pl.List(pl.String),
    "title": pl.String,
    "message": pl.String,
    "from_to": pl.List(pl.Struct({"from": stop_struct, "to": stop_struct})),
}
df = pl.DataFrame(disruptions, schema=schema)

output_filename = os.path.join(
    OUTPUT_DIR, f"traffic-{now.year}-{now.month:02}-{now.day:02}.parquet"
)

if os.path.isfile(output_filename):
    old_df = pl.scan_parquet(output_filename)

    df = (
        pl.concat((df.lazy(), old_df), how="vertical", rechunk=True)
        .unique(subset=["id"], keep="last", maintain_order=True)
        .collect()
    )

print(len(df))
df.write_parquet(output_filename)
