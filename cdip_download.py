from pathlib import Path

import requests
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# NAGS HEAD, NC - 243
# station_number = "243"
# KANEOHE BAY, WETS, HI - 225
station_number = "225"
# 1.28 hz * 30
SAMPLES_PER_HALF_HOUR = 2304


def get_cdip_displacement_df(station_number, dataset_number):
    fname = f"{station_number}p1_d{dataset_number}.nc"

    nc_path = Path(f"./data/00_raw/{fname}").resolve()
    print(f"Opening {nc_path} if it exists...")

    if nc_path.exists() is False:
        nc_url = f"https://thredds.cdip.ucsd.edu/thredds/fileServer/cdip/archive/{station_number}p1/{fname}"
        print("Downloading", nc_url)
        # Download the NetCDF file using requests
        response = requests.get(nc_url)
        with open(nc_path, "wb") as f:
            f.write(response.content)

    # Open the downloaded NetCDF file with xarray
    ds = xr.open_dataset(nc_path)

    # Extract the relevant variables from the dataset
    xdisp = ds["xyzXDisplacement"]  # North/South Displacement (X)
    ydisp = ds["xyzYDisplacement"]  # East/West Displacement (Y)
    zdisp = ds["xyzZDisplacement"]  # Vertical Displacement (Z)
    qc_flag = ds["xyzFlagPrimary"]  # Quality control flag

    # For some reason all of these are missing one sample. So we remove the last section

    xdisp = xdisp[:-(SAMPLES_PER_HALF_HOUR)]
    ydisp = ydisp[:-(SAMPLES_PER_HALF_HOUR)]
    zdisp = zdisp[:-(SAMPLES_PER_HALF_HOUR)]
    qc_flag = qc_flag[:-(SAMPLES_PER_HALF_HOUR)]

    filter_delay = ds["xyzFilterDelay"].values
    start_time = ds["xyzStartTime"].values  # Start time of buoy data collection
    sample_rate = float(
        ds["xyzSampleRate"].values
    )  # Sample rate of buoy data collection
    sample_rate = round(sample_rate, 2)
    print(
        f"Station Number: {station_number}, dataset_number: {dataset_number}, sample_rate: {sample_rate}"
    )

    print(f"Len xdisp: {len(xdisp)}, num 30 min sections = {(len(xdisp) + 1) / 2304}")
    print(f"Filter delay: {filter_delay}")

    sample_delta_t_seconds = 1 / sample_rate
    sample_delta_t_nanoseconds = sample_delta_t_seconds * 1e9
    n_times = len(xdisp)

    start_time_ns = start_time.astype("int64")

    start_time_ns = start_time.astype("int64")  # Convert start_time to nanoseconds
    # start_time_ns -= filter_delay * 1e9
    time_increments = (
        np.arange(n_times) * sample_delta_t_nanoseconds
    )  # Create an array of time increments
    times = start_time_ns + time_increments

    time = pd.to_datetime(times, unit="ns", origin="unix", utc=True)  # type: ignore

    df = pd.DataFrame(
        {
            "north_displacement_meters": xdisp,
            "east_displacement_meters": ydisp,
            "vert_displacement_meters": zdisp,
            "qc_displacement": qc_flag,
        },
        index=time,
    )

    return df


station_number = "225"
station_number = "243"
df_1 = get_cdip_displacement_df(station_number, "01")
# print(df_1.info())
# print(df_1.head())
# print(df_1.tail())
df_2 = get_cdip_displacement_df(station_number, "02")
# print(df_2.info())
# print(df_2.head())
# print(df_2.tail())
df_3 = get_cdip_displacement_df(station_number, "03")
# print(df_3.info())
# print(df_3.head())
# print(df_3.tail())
df_4 = get_cdip_displacement_df(station_number, "04")
# df_5 = get_cdip_displacement_df(station_number, "05")

# df_all = pd.concat([df_1, df_2, df_3, df_4, df_5], axis="index")
df_all = pd.concat([df_1, df_2, df_3, df_4], axis="index")
# df_all = pd.concat([df_1, df_2, df_3], axis="index")
df_all = df_all.sort_index()

df_all.to_parquet(f"./data/a1_one_to_one_parquet/{station_number}_all.parquet")

print(df_all.info())
print(df_all.head())
print(df_all.tail())

print(df_all.describe())

print(f"Successfully saved {station_number}_all.parquet!")
