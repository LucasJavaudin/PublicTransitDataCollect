# Public Transit Data Collect

This repository contains the Python scripts that I am using to collect public transit timetables
data from [transport.data.gouv.fr](https://transport.data.gouv.fr/).

## Use

- Install the required Python packages listed in `requirements.txt`.
- Run the script `gtfs_to_parquet.py` at regular time interval (e.g., each day) to collect public
  transit timetables from all datasets at [transport.data.gouv.fr](https://transport.data.gouv.fr/)
  as Parquet files in the `data/` directory.
