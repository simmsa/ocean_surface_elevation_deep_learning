














































import folium

def create_location_map(latitude, longitude, label, zoom_level=8):
    location_map = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)

    folium.Marker(
        [latitude, longitude],
        popup=label,
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(location_map)

    return location_map



#| label: fig-wave-wets-map
#| fig-cap: Map of CDIP Buoy 225 Location at the Wave Energy Test Site, Kaneohe Bay, Oahu, HI

wets_map = create_location_map(
    latitude=21.47740,
    longitude=-157.75684,
    label='CDIP 225 - WETS Hawaii'
)
wets_map






#| label: fig-wave-nags-head-map
#| fig-cap: Map of CDIP Buoy 243 Location in Nags Head, NC

nags_head_map = create_location_map(
    latitude= 36.00150,
    longitude=-75.42090,
    label='CDIP 243 - Nags Head NC'
)
nags_head_map
































#| label: fig-buoy-movement
#| fig-cap: Directional Reference Frame of Datawell Waverider DWR-MkIII Buoy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
sns.set_theme()

# Get a pleasing color palette
# colors = sns.color_palette("husl", 3)  # Using husl for distinct but harmonious colors
colors = sns.color_palette()
x_color = colors[0]
y_color = colors[1]
z_color = colors[2]

# Create figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Create sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 0.4 * np.outer(np.cos(u), np.sin(v))
y = 0.4 * np.outer(np.sin(u), np.sin(v))
z = 0.4 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot semi-transparent sphere
ax.plot_surface(x, y, z, color='orange', alpha=0.3)

# Plot axes through sphere center
length = 0.6
ax.plot([-length, length], [0, 0], [0, 0], color=x_color, linewidth=2, label='X (East/West)')
ax.plot([0, 0], [-length, length], [0, 0], color=y_color, linewidth=2, label='Y (True North/South)')
ax.plot([0, 0], [0, 0], [-length, length], color=z_color, linewidth=2, label='Z (Vertical)')

# Add arrows at the ends
arrow_length = 0.1
# X axis arrows
ax.quiver(length, 0, 0, arrow_length, 0, 0, color=x_color, arrow_length_ratio=0.3)
ax.quiver(-length, 0, 0, -arrow_length, 0, 0, color=x_color, arrow_length_ratio=0.3)
# Y axis arrows
ax.quiver(0, length, 0, 0, arrow_length, 0, color=y_color, arrow_length_ratio=0.3)
ax.quiver(0, -length, 0, 0, -arrow_length, 0, color=y_color, arrow_length_ratio=0.3)
# Z axis arrows
ax.quiver(0, 0, length, 0, 0, arrow_length, color=z_color, arrow_length_ratio=0.3)
ax.quiver(0, 0, -length, 0, 0, -arrow_length, color=z_color, arrow_length_ratio=0.3)


# Set equal aspect ratio
ax.set_box_aspect([1,1,1])

# Set axis limits
limit = 0.55
ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

# Add grid
ax.grid(True, alpha=0.3)

# Add axis labels with matching colors
ax.set_xlabel('East/West Displacement (X) [m]', color=x_color, weight='bold', fontsize=18)
ax.set_ylabel('True North/South Displacement (Y) [m]', color=y_color, weight='bold', fontsize=18)
ax.set_zlabel('Vertical Displacement (Z) [m]', color=z_color, weight='bold', fontsize=18)

# Adjust view angle
ax.view_init(elev=20, azim=180 - 45)

# Set background color to white
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.show()


































































#| label: fig-displacement
#| fig-cap: "CDIP 225 30 minutes of Displacement - November 11, 2017 @ 11 am. Data from @cdip_wets"
#| fig-subcap:
#|   - Vertical (Z) Displacement
#|   - North/South (Y) Displacement
#|   - East/West (X) Displacement
#| layout-nrow: 3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
df = pd.read_parquet("../data/a2_std_partition/station_number=0225/year=2017/month=11/day=11/hour=11/minute=00/data_20171111_1100.parquet")

df['vert_displacement_meters'].plot(figsize=(9, 2), linewidth=0.85, xlabel="Time", ylabel="Vertical\nDisplacement [m]")
plt.show()

df['north_displacement_meters'].plot(figsize=(9, 2), linewidth=0.85, xlabel="Time", ylabel="North/South\nDisplacement [m]")
plt.show()

df['east_displacement_meters'].plot(figsize=(9, 2), linewidth=0.85, xlabel="Time", ylabel="East/West\nDisplacement [m]")
plt.show()





#| label: fig-displacement-zoomed
#| fig-cap: "CDIP 225 1.5 minutes of Displacement - November 11, 2017 @ 11 am. Data from @cdip_wets"
#| fig-subcap:
#|   - Vertical (Z) Displacement
#|   - North/South (Y) Displacement
#|   - East/West (X) Displacement
#| layout-nrow: 3

sns.set_theme()
df = pd.read_parquet("../data/a2_std_partition/station_number=0225/year=2017/month=11/day=11/hour=11/minute=00/data_20171111_1100.parquet")

end_index = int(288 / 2) # 2304 / 2 / 2 / 2 - ~3 minutes / 2 = 90 seconds

df['vert_displacement_meters'].iloc[:end_index].plot(figsize=(9, 2), linewidth=0.85, xlabel="Time", ylabel="Vertical\nDisplacement [m]", marker=".", markersize=2)
plt.show()

df['north_displacement_meters'].iloc[:end_index].plot(figsize=(9, 2), linewidth=0.85, xlabel="Time", ylabel="North/South\nDisplacement [m]", marker=".", markersize=2)
plt.show()
df['east_displacement_meters'].iloc[:end_index].plot(figsize=(9, 2), linewidth=0.85, xlabel="Time", ylabel="East/West\nDisplacement [m]", marker=".", markersize=2)
plt.show()























#| eval: false
#| echo: true
#| code-fold: false
#| lst-label: lst-cdip-download
#| lst-cap: CDIP Download Implementation

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
df_2 = get_cdip_displacement_df(station_number, "02")
df_3 = get_cdip_displacement_df(station_number, "03")
df_4 = get_cdip_displacement_df(station_number, "04")
# df_5 = get_cdip_displacement_df(station_number, "05")

# df_all = pd.concat([df_1, df_2, df_3, df_4, df_5], axis="index")
df_all = pd.concat([df_1, df_2, df_3, df_4], axis="index")
# df_all = pd.concat([df_1, df_2, df_3], axis="index")
df_all = df_all.sort_index()

df_all.to_parquet(f"./data/a1_one_to_one_parquet/{station_number}_all.parquet")

print(df_all.info())

print(f"Successfully saved {station_number}_all.parquet!")















first_225 = pd.read_parquet("../data/a2_std_partition/station_number=0225/year=2016/month=08/day=26/hour=22/minute=00/data_20160826_2200.parquet")
first_225_timestamp = first_225.index[0]
first_225_timestamp
last_225 = pd.read_parquet("../data/a2_std_partition/station_number=0225/year=2024/month=09/day=11/hour=18/minute=30/data_20240911_1830.parquet")
last_225_timestamp = last_225.index[-1]
last_225_timestamp
first_243 = pd.read_parquet("../data/a2_std_partition/station_number=0243/year=2018/month=08/day=26/hour=15/minute=00/data_20180826_1500.parquet")
first_243_timestamp = first_243.index[0]
first_243_timestamp

last_243 = pd.read_parquet("../data/a2_std_partition/station_number=0243/year=2023/month=07/day=12/hour=23/minute=30/data_20230712_2330.parquet")
last_243_timestamp = last_243.index[-1]
last_243_timestamp

from datetime import datetime

# Create the data
data = {
    'station': ['225', '243'],
    'start_date': [
        first_225_timestamp,
        first_243_timestamp,
    ],
    'end_date': [
        last_225_timestamp,
        last_243_timestamp,
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate duration
df['duration'] = df['end_date'] - df['start_date']

# Function to format duration in human readable format
def format_duration(timedelta):
    years = timedelta.days // 365
    remaining_days = timedelta.days % 365
    months = remaining_days // 30
    days = remaining_days % 30

    parts = []
    if years > 0:
        parts.append(f"{years} {'year' if years == 1 else 'years'}")
    if months > 0:
        parts.append(f"{months} {'month' if months == 1 else 'months'}")
    if days > 0:
        parts.append(f"{days} {'day' if days == 1 else 'days'}")

    return ", ".join(parts)

# Add human readable duration
df['duration_human'] = df['duration'].apply(format_duration)

# Format datetime columns to be more readable
df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d %H:%M')
df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d %H:%M')

df = df.rename({
    'start_date': "Start Date [UTC]",
    'end_date': "End Date [UTC]",
    'duration_human': "Duration",
}, axis="columns")






#| label: tbl-duration
#| tbl-cap: Temporal Details of Downloaded CDIP Data


df[['Start Date [UTC]', 'End Date [UTC]', 'Duration']]













import duckdb
import os

def calculate_column_stats(partition_path, column_name):
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false;")

    query = f"""
    SELECT
        '{column_name}' as column_name,
        COUNT({column_name}) as count,
        COUNT(DISTINCT {column_name}) as unique_count,
        SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_count,
        MIN({column_name}) as min_value,
        MAX({column_name}) as max_value,
        AVG({column_name}::DOUBLE) as mean,
        STDDEV({column_name}::DOUBLE) as std_dev,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column_name}::DOUBLE) as q1,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name}::DOUBLE) as median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column_name}::DOUBLE) as q3
    FROM read_parquet('{partition_path}/**/*.parquet', hive_partitioning=true)
    WHERE {column_name} IS NOT NULL
    """

    stats_df = con.execute(query).df()
    con.close()
    return stats_df

def analyze_displacement_data(base_path, columns_to_analyze, station_numbers, output_path, overwrite=False):
    # Check if stats file already exists
    if os.path.exists(output_path) and not overwrite:
        return pd.read_parquet(output_path)

    all_stats = []

    for station in station_numbers:
        station_str = f"{station:04d}"  # Format station number with leading zeros
        partition_path = f"{base_path}/station_number={station_str}"

        if not os.path.exists(partition_path):
            print(f"Skipping station {station_str} - path does not exist")
            continue

        print(f"Processing station {station_str}...")

        for column in columns_to_analyze:
            try:
                stats_df = calculate_column_stats(partition_path, column)
                stats_df['station'] = station_str
                all_stats.append(stats_df)
                print(f"  Completed analysis of {column}")
            except Exception as e:
                print(f"  Error processing {column} for station {station_str}: {str(e)}")

    # Combine all results
    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to parquet
        combined_stats.to_parquet(output_path, index=False)
        print(f"\nStatistics saved to {output_path}")
        return combined_stats
    else:
        print("No statistics were generated")
        return None

# Example usage
base_path = "../data/a2_std_partition"
columns_to_analyze = [
    "vert_displacement_meters",
    "north_displacement_meters",
    "east_displacement_meters"
]
station_numbers = [225, 243]  # Add more station numbers as needed
output_path = "../data/displacement_stats.parquet"

# Run the analysis - will load existing file if it exists
stats_df = analyze_displacement_data(
    base_path=base_path,
    columns_to_analyze=columns_to_analyze,
    station_numbers=station_numbers,
    output_path=output_path,
    overwrite=False  # Set to True to force recalculation
)

stats_df["Range [m]"] = stats_df['max_value'] + stats_df['min_value'].abs()

# stats_df.loc[stats_df['column_name'] == 'vert_displacement_meters'] = "Vertical Displacement [m]"
stats_df.loc[stats_df['column_name'] == 'vert_displacement_meters', 'column_name'] = "Vertical Displacement [m]"
stats_df.loc[stats_df['column_name'] == 'north_displacement_meters', 'column_name'] = "North/South Displacement [m]"
stats_df.loc[stats_df['column_name'] == 'east_displacement_meters', 'column_name'] = "East/West Displacement [m]"

stats_df.loc[stats_df['station'] == '0225', 'station'] = "225 - Kaneohe Bay, HI"
stats_df.loc[stats_df['station'] == '0243', 'station'] = "243 - Nags Head, NC"
# stats_df = stats_df.rename(columns={'vert_displacement_meters': 'Vertical Displacement [m]'})

stats_df = stats_df.rename({
    "count": "Count",
    "min_value": "Min [m]",
    "max_value": "Max [m]",
    "mean": "Mean [m]",
    "std_dev": "Standard Deviation [m]",
}, axis="columns")

# stats_df



import matplotlib

matplotlib.rcParams["axes.formatter.limits"] = (-99, 99)


def plot_stat(this_stat_df, stat_column, decimals=2):
    plt.figure(figsize=(9, 3))
    plt.gca().yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax = sns.barplot(this_stat_df, x="station", y=stat_column, hue="column_name")

    for container in ax.containers:
        ax.bar_label(container, fmt=f"%.{decimals}f", padding=3, fontsize=7)

    plt.xlabel(None)
    # Move legend below the plot, set to 3 columns, remove box
    plt.legend(
        bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, frameon=False, title=""
    )





#| label: fig-stats-count
#| fig-cap: "Available Data: Sample Count"

plot_stat(stats_df, 'Count', decimals=0)



#| label: fig-stats-range
#| fig-cap: "Available Data: Range [m]"

plot_stat(stats_df, 'Range [m]')



#| label: fig-stats-max
#| fig-cap: "Available Data: Maximum [m]"

plot_stat(stats_df, 'Max [m]')



#| label: fig-stats-min
#| fig-cap: "Available Data: Minimum [m]"

plot_stat(stats_df, 'Min [m]')



#| label: fig-stats-std
#| fig-cap: "Available Data: Standard Deviation [m]"

plot_stat(stats_df, 'Standard Deviation [m]')































#| code-fold: false
#| lst-label: lst-partition
#| lst-cap: Data Partitioning Implementation


def partition_df(this_df, station_number, output_folder):
    # Ensure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Sort the DataFrame by index (assuming timestamp index)
    this_df = this_df.sort_index()

    # Function to create partition path
    def create_partition_path(timestamp, station):
        return (
            f"station_number={station:04d}/"
            f"year={timestamp.year:04d}/"
            f"month={timestamp.month:02d}/"
            f"day={timestamp.day:02d}/"
            f"hour={timestamp.hour:02d}/"
            f"minute={timestamp.minute:02d}"
        )

    # Process data in chunks of 2304 samples
    chunk_size = 2304
    num_chunks = len(this_df) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        # Get chunk of data
        chunk_df = this_df.iloc[start_idx:end_idx]

        # Verify chunk duration
        chunk_duration = chunk_df.index[-1] - chunk_df.index[0]
        expected_duration = timedelta(minutes=30)

        # Use start time of chunk for partitioning
        chunk_start_time = chunk_df.index[0]

        # Create partition path
        partition_path = create_partition_path(chunk_start_time, station_number)
        full_path = output_folder / partition_path

        # Create directory structure
        full_path.mkdir(parents=True, exist_ok=True)

        # Save the partitioned data
        output_file = (
            full_path / f"data_{chunk_start_time.strftime('%Y%m%d_%H%M')}.parquet"
        )
        chunk_df.to_parquet(output_file)

        print(f"Saved partition: {partition_path}")
        print(f"Chunk {i} duration: {chunk_duration}")

    # Handle any remaining data
    if len(this_df) % chunk_size != 0:
        remaining_df = this_df.iloc[num_chunks * chunk_size :]
        print(f"Warning: {len(remaining_df)} samples remaining at end of file")
        print(f"Last timestamp: {remaining_df.index[-1]}")



















#| eval: false
#| echo: true
#| code-fold: false
#| lst-label: lst-mhkit
#| lst-cap: Calculation of Wave Quantities of Interest from Displacement

import mhkit.wave as wave

def calculate_wave_qoi(input_df, station_number, path):
    input_df = input_df.dropna(axis="index")
    column = "vert_displacement_meters"

    if len(input_df) != 2304:
        return None

    sample_rate_hz = 1.28
    n_fft = 256
    window = "hann"
    detrend = True
    surface_elevation = pd.DataFrame(input_df[column].iloc[:2048])
    spectra = wave.resource.elevation_spectrum(
        surface_elevation, sample_rate_hz, n_fft, window=window, detrend=detrend
    )

    return {
        "time": input_df.index[0],
        "significant_wave_height_meters": wave.resource.significant_wave_height(spectra)["Hm0"].to_list()[0],  # type: ignore
        "energy_period_seconds": wave.resource.energy_period(spectra)["Te"].to_list()[0],  # type: ignore
        "omnidirectional_wave_energy_flux": wave.resource.energy_flux(spectra, np.nan, deep=True)["J"].to_list()[0],  # type: ignore
        "station_number": station_number,
        "path": str(path),
    }





qoi_225 = pd.read_parquet("../data/b2_wave_qoi_stats/qoi_225.parquet")
qoi_243 = pd.read_parquet("../data/b2_wave_qoi_stats/qoi_243.parquet")






#| label: fig-wave-stats-hm0
#| fig-cap: "Significant Wave Height, $H_{m_0}$ [$m$]"
#| fig-subcap:
#|   - "Wave Energy Test Site - Hawaii"
#|   - "Nags Head - North Carolina"
#| layout-ncol: 2

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def compare_qoi_over_time(qoi, label):
    figsize = (8, 3)
    xlabel = "Time"
    qoi_225[qoi].plot(
        figsize=figsize, xlabel=xlabel, ylabel=label, linewidth=0.75
    )
    plt.show()

    qoi_243[qoi].plot(
        figsize=figsize, xlabel=xlabel, ylabel=label, linewidth=0.75
    )
    plt.show()

compare_qoi_over_time('significant_wave_height_meters', "$H_{m_0}$ [$m$]")



#| label: fig-wave-stats-te
#| fig-cap: "Energy Period, $T_e$ [$s$]"
#| fig-subcap:
#|   - "Wave Energy Test Site - Hawaii"
#|   - "Nags Head - North Carolina"
#| layout-ncol: 2

compare_qoi_over_time('energy_period_seconds', "$T_e$ [$s$]")



#| label: fig-wave-stats-j
#| fig-cap: "Omnidirectional Wave Energy Flux (Wave Power), $J$ [$W/m$]"
#| fig-subcap:
#|   - "Wave Energy Test Site - Hawaii"
#|   - "Nags Head - North Carolina"
#| layout-ncol: 2

compare_qoi_over_time('omnidirectional_wave_energy_flux', "$J$ [$W/m$]")







#| label: fig-wave-stats-hm0-2019
#| fig-cap: "Significant Wave Height, $H_{m_0}$ [$m$]"
#| fig-subcap:
#|   - "Wave Energy Test Site - Hawaii"
#|   - "Nags Head - North Carolina"
#| layout-ncol: 2

start_date = "2019-01-01 00:00:00"
end_date = "2019-12-31 23:59:59.9999"

qoi_225 = qoi_225.loc[start_date:end_date]
qoi_243 = qoi_243.loc[start_date:end_date]


compare_qoi_over_time('significant_wave_height_meters', "$H_{m_0}$ [$m$]")



#| label: fig-wave-stats-te-2019
#| fig-cap: "Energy Period, $T_e$ [$s$]"
#| fig-subcap:
#|   - "Wave Energy Test Site - Hawaii"
#|   - "Nags Head - North Carolina"
#| layout-ncol: 2

compare_qoi_over_time('energy_period_seconds', "$T_e$ [$s$]")



#| label: fig-wave-stats-j-2019
#| fig-cap: "Omnidirectional Wave Energy Flux (Wave Power), $J$ [$W/m$]"
#| fig-subcap:
#|   - "Wave Energy Test Site - Hawaii"
#|   - "Nags Head - North Carolina"
#| layout-ncol: 2

compare_qoi_over_time('omnidirectional_wave_energy_flux', "$J$ [$W/m$]")





















import numpy as np

def plot_wave_heatmap(df, figsize=(12, 8)):
    # Create bins for Hm0 and Te
    hm0_bins = np.arange(0, df['significant_wave_height_meters'].max() + 0.5, 0.5)
    te_bins = np.arange(0, df['energy_period_seconds'].max() + 1, 1)

    # Use pd.cut to bin the data
    hm0_binned = pd.cut(df['significant_wave_height_meters'],
                        bins=hm0_bins,
                        labels=hm0_bins[:-1],
                        include_lowest=True)

    te_binned = pd.cut(df['energy_period_seconds'],
                       bins=te_bins,
                       labels=te_bins[:-1],
                       include_lowest=True)

    # Create cross-tabulation of binned data
    counts = pd.crosstab(hm0_binned, te_binned)

    counts = counts.sort_index(ascending=False)


    # Replace 0 counts with NaN
    counts = counts.replace(0, np.nan)

    # Create figure and axis
    plt.figure(figsize=figsize)

    # Create heatmap using seaborn
    ax = sns.heatmap(
        counts,
        cmap='viridis',
        annot=True,  # Add count annotations
        fmt='.0f',   # Format annotations as integers
        cbar_kws={'label': 'Count'},
    )

    # Customize plot
    plt.xlabel('Energy Period Te (s)')
    plt.ylabel('Significant Wave Height Hm0 (m)')

    # Rotate x-axis labels for better readability
    # plt.xticks(rotation=45)
    plt.yticks(rotation=90)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()




qoi_225 = pd.read_parquet("../data/b2_wave_qoi_stats/qoi_225.parquet")
qoi_243 = pd.read_parquet("../data/b2_wave_qoi_stats/qoi_243.parquet")





#| label: fig-wave-stats-225-count
#| fig-cap: "Distribution of Sea States at WETS (CDIP 225), Showing Occurrence Count of Combined Significant Wave Height ($H_{m_0}$) and Energy Period ($T_e$) Conditions"

plot_wave_heatmap(qoi_225)



#| label: fig-wave-stats-243-count
#| fig-cap: "Distribution of Sea States at Nags Head (CDIP 243), Showing Occurrence Count of Combined Significant Wave Height ($H_{m_0}$) and Energy Period ($T_e$) Conditions"

plot_wave_heatmap(qoi_243)






























#| code-fold: false
#| lst-label: lst-bin-sampling
#| lst-cap: Function to Sample Sea State Matrix Bins

def sample_wave_bins(df, n_samples=1, hm0_step=0.5, te_step=1.0):
    # Create bins for Hm0 and Te
    hm0_bins = np.arange(0, df['significant_wave_height_meters'].max() + hm0_step, hm0_step)
    te_bins = np.arange(0, df['energy_period_seconds'].max() + te_step, te_step)

    # Add bin columns to the dataframe
    df_binned = df.copy()
    df_binned['hm0_bin'] = pd.cut(df['significant_wave_height_meters'],
                                 bins=hm0_bins,
                                 labels=hm0_bins[:-1],
                                 include_lowest=True)

    df_binned['te_bin'] = pd.cut(df['energy_period_seconds'],
                                bins=te_bins,
                                labels=te_bins[:-1],
                                include_lowest=True)

    # Convert category types to float
    df_binned['hm0_bin'] = df_binned['hm0_bin'].astype(float)
    df_binned['te_bin'] = df_binned['te_bin'].astype(float)

    # Sample from each bin combination
    samples = []
    for hm0_val in df_binned['hm0_bin'].unique():
        for te_val in df_binned['te_bin'].unique():
            bin_data = df_binned[
                (df_binned['hm0_bin'] == hm0_val) &
                (df_binned['te_bin'] == te_val)
            ]

            if not bin_data.empty:
                # Sample min(n_samples, bin size) rows from this bin
                bin_samples = bin_data.sample(
                    n=min(n_samples, len(bin_data)),
                    random_state=42  # For reproducibility
                )
                samples.append(bin_samples)

    # Combine all samples
    if samples:
        result = pd.concat(samples, axis=0).reset_index(drop=True)

        # Add bin center values for reference
        result['hm0_bin_center'] = result['hm0_bin'] + (hm0_step / 2)
        result['te_bin_center'] = result['te_bin'] + (te_step / 2)
        result.insert(0, 'station_number', result.pop('station_number'))


        return result
    else:
        return pd.DataFrame()







data_225 = sample_wave_bins(qoi_225)
data_225.to_parquet("../model_input_spec_225.parquet")



data_225.head()



data_225.info()



data_243 = sample_wave_bins(qoi_243)
data_243.to_parquet("../model_input_spec_243.parquet")



data_243.head()



data_243.info()
















n_samples = 5
data_225 = sample_wave_bins(qoi_225, n_samples=n_samples)
data_225.to_parquet(f"../model_input.samples_{n_samples}._spec_225.parquet")

data_243 = sample_wave_bins(qoi_243, n_samples=n_samples)
data_243.to_parquet(f"../model_input.samples_{n_samples}.spec_243.parquet")



print(len(data_225))



print(len(data_243))

















#| eval: false
#| echo: true
#| code-fold: false
#| lst-label: lst-lstm-model
#| lst-cap: Long Short-Term Memory PyTorch Model

class LSTMModel(WavePredictionModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()  # Save all init parameters

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out)
        return predictions










#| eval: false
#| echo: true
#| code-fold: false
#| lst-label: lst-enh-lstm-model
#| lst-cap: Enhanced Long Short-Term Memory PyTorch Model

class EnhancedLSTMModel(WavePredictionModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        bidirectional: bool = True,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # Main LSTM layers with skip connections
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        lstm_input_dim = hidden_dim
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        for _ in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    lstm_input_dim,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=0,
                )
            )
            self.layer_norms.append(nn.LayerNorm(lstm_output_dim))
            lstm_input_dim = lstm_output_dim

        # Output processing
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Linear(hidden_dim // 2, input_dim),
            ]
        )

        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(len(self.output_layers))]
        )

        # Skip connection
        self.skip_connection = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original input for skip connection
        original_input = x

        # Input processing
        x = self.input_layer(x)

        # Process through LSTM layers with residual connections
        for lstm, norm in zip(self.lstm_layers, self.layer_norms):
            residual = x
            x, _ = lstm(x)
            x = norm(x)
            if residual.shape == x.shape:
                x = x + residual

        # Output processing
        for linear, dropout in zip(self.output_layers[:-1], self.dropouts[:-1]):
            residual = x
            x = linear(x)
            x = F.relu(x)
            x = dropout(x)
            if residual.shape == x.shape:
                x = x + residual

        # Final output layer
        x = self.output_layers[-1](x)

        # Add skip connection
        x = x + self.skip_connection(original_input)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)












#| eval: false
#| echo: true
#| code-fold: false
#| lst-label: lst-trans-model
#| lst-cap: Transformer PyTorch Model

class TransformerModel(WavePredictionModel):

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ):
        super().__init__(input_dim, learning_rate)
        self.save_hyperparameters()  # Save all init parameters

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        return self.output_projection(x)






































#| label: fig-225-train-baseline
#| fig-cap: "Baseline Model: Training vs. Validation Mean Absolute Error"
#| fig-subcap:
#|   - "CDIP 225"
#|   - "CDIP 243"
#| layout-nrow: 2

df = pd.read_parquet("../training_history/training_history_lstm_20241207_205254.parquet")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()

df = pd.read_parquet("../training_history/training_history_lstm_20241207_215544.parquet")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()











#| label: fig-225-train-100-LSTM
#| fig-cap: "100 Epoch LSTM Model: Training vs. Validation Mean Absolute Error"
#| fig-subcap:
#|   - "CDIP 225"
#|   - "CDIP 243"
#| layout-nrow: 2

df = pd.read_parquet("../training_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()

df = pd.read_parquet("../training_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()










#| label: fig-225-train-transformer
#| fig-cap: "Transformer Model: Training vs. Validation Mean Absolute Error"
#| fig-subcap:
#|   - "CDIP 225"
#|   - "CDIP 243"
#| layout-nrow: 2

df = pd.read_parquet("../training_history/model_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])

plt.show()

df = pd.read_parquet("../training_history/model_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/")
df = df.set_index(['epoch'])
fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])

plt.show()











#| label: fig-train-4-layer-LSTM
#| fig-cap: "4 Layer LSTM Model: Training vs. Validation Mean Absolute Error"
#| fig-subcap:
#|   - "CDIP 225"
#|   - "CDIP 243"
#| layout-nrow: 2

df = pd.read_parquet("../training_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()

df = pd.read_parquet("../training_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()










#| label: fig-train-6-layer-LSTM
#| fig-cap: "6 Layer LSTM Model: Training vs. Validation Mean Absolute Error"
#| fig-subcap:
#|   - "CDIP 225"
#|   - "CDIP 243"
#| layout-nrow: 2

df = pd.read_parquet("../training_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])

plt.show()

df = pd.read_parquet("../training_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()











#| label: fig-train-enhanced-trans
#| fig-cap: "Enhanced Transformer Model: Training vs. Validation Mean Absolute Error"
#| fig-subcap:
#|   - "CDIP 225"
#|   - "CDIP 243"
#| layout-nrow: 2

df = pd.read_parquet("../training_history/model_enhanced_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])

plt.show()

df = pd.read_parquet("../training_history/model_enhanced_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])
plt.show()










#| label: fig-train-enhanced-lstm
#| fig-cap: "Enhanced LSTM Model: Training vs. Validation Mean Absolute Error"
#| fig-subcap:
#|   - "CDIP 225"
#|   - "CDIP 243"
#| layout-nrow: 2

df = pd.read_parquet("../training_history/model_enhanced_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])

plt.show()

df = pd.read_parquet("../training_history/model_enhanced_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/")
df = df.set_index(['epoch'])

fig, axs = plt.subplots(figsize=(12, 2))
df[['train_mae', 'val_mae']].plot(ax=axs, linewidth = 0.85, xlabel="Epoch", ylabel="Mean Absolute Error")
axs.legend(['Train', "Validate"])





















result_stats = []



from sklearn.metrics import mean_absolute_error, r2_score

def calc_stats(label, station, targets, predictions, column='vert_displacement_meters'):
    return {
		'label': label,
		'station': station,
        'mae': mean_absolute_error(targets[column], predictions[column]),
        'r2': r2_score(targets[column], predictions[column]),
        'correlation': np.corrcoef(targets[column], predictions[column])[0,1]
    }




def plot_test_section_compared_to_input(index, this_bins_df, this_source, this_targets, this_predictions, n_samples=128):
    # Calculate start and stop indices
    start = index * n_samples
    stop = start + n_samples

    # Get source path and load input data
    source_path = this_source.iloc[index]['Source Path']
    input_df = pd.read_parquet(source_path)

    # Get statistics for the title
    stats = this_bins_df[this_bins_df["path"] == source_path]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 3))

    # Create index arrays for proper alignment
    input_index = np.arange(n_samples * 2)
    shifted_index = np.arange(n_samples, 2 * n_samples)

    # Plot input data with original index
    ax.plot(input_index, input_df['vert_displacement_meters'].iloc[:n_samples * 2].values,
            linewidth=0.85, label="Input", alpha=0.7)

    # Plot target and prediction with shifted index
    ax.plot(shifted_index,
            this_targets['vert_displacement_meters'].iloc[start:stop].values,
            label="Target", linewidth=0.85)
    ax.plot(shifted_index,
            this_predictions['vert_displacement_meters'].iloc[start:stop].values,
            label="Prediction", linewidth=0.75)

    # Configure plot
    plt.ylabel("Vertical Displacement [m]")
    plt.legend(loc="upper right")
    plt.title(
        f"CDIP {stats['station_number'].item()} - $H_{{m0}}$: {stats['hm0_bin'].item()}, "
        f"$T_{{e}}$: {stats['te_bin'].item()}"
    )

    plt.tight_layout()
    plt.show()

def plot_test_section(index, this_bins_df, this_source, this_targets, this_predictions, n_samples=128):
    # Calculate start and stop indices
    start = index * n_samples
    stop = start + n_samples

    scale_factor = 0.5

    this_targets = this_targets.copy()
    this_predictions = this_predictions.copy()

    this_targets *= scale_factor
    this_predictions *= scale_factor

    # Get source path and load input data
    source_path = this_source.iloc[index]['Source Path']

    # Get statistics for the title
    stats = this_bins_df[this_bins_df["path"] == source_path]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 3))

    ax.plot(
            this_targets['vert_displacement_meters'].iloc[start:stop].values,
            label="Target", linewidth=0.85, marker=".", markersize=4)
    ax.plot(
            this_predictions['vert_displacement_meters'].iloc[start:stop].values,
            label="Prediction", linewidth=0.75, marker=".", markersize=4)

    # Configure plot
    plt.ylabel("Vertical Displacement [m]")
    plt.legend(loc="upper right")
    plt.title(
        f"CDIP {stats['station_number'].item()} - $H_{{m_0}}$: {stats['hm0_bin'].item()}, "
        f"$T_{{e}}$: {stats['te_bin'].item()}",
        fontsize=18,
    )

    plt.tight_layout()
    plt.show()

















df_bins = pd.read_parquet("./model_input_spec_225.parquet")
targets = pd.read_parquet("../testing_history/model_lstm.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.lstm_20241207_205411.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.lstm_20241207_205411.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.lstm_20241207_205411.parquet")

result_stats.append(calc_stats("Baseline", "225", targets, predictions))



#| label: fig-lstm-baseline-test-section-1
#| fig-cap: Baseline LSTM - CDIP 225 - Test Section 1

plot_test_section(0, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-2
#| fig-cap: Baseline LSTM - CDIP 225 - Test Section 2

plot_test_section(1, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-3
#| fig-cap: Baseline LSTM - CDIP 225 - Test Section 3

plot_test_section(2, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-4
#| fig-cap: Baseline LSTM - CDIP 225 - Test Section 4

plot_test_section(3, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-5
#| fig-cap: Baseline LSTM - CDIP 225 - Test Section 5

plot_test_section(4, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-6
#| fig-cap: Baseline LSTM - CDIP 225 - Test Section 6

plot_test_section(5, df_bins, sources, targets, predictions)









targets = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.lstm_20241207_215743.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.lstm_20241207_215743.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.lstm_20241207_215743.parquet")
df_bins = pd.read_parquet("./model_input_spec_243.parquet")

result_stats.append(calc_stats("Baseline", "243", targets, predictions))



#| label: fig-lstm-baseline-test-section-1-nags
#| fig-cap: Baseline LSTM - CDIP 243 - Test Section 1

plot_test_section(0, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-2-nags
#| fig-cap: Baseline LSTM - CDIP 243 - Test Section 2

plot_test_section(1, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-3-nags
#| fig-cap: Baseline LSTM - CDIP 243 - Test Section 3

plot_test_section(2, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-4-nags
#| fig-cap: Baseline LSTM - CDIP 243 - Test Section 4

plot_test_section(3, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-5-nags
#| fig-cap: Baseline LSTM - CDIP 243 - Test Section 5

plot_test_section(4, df_bins, sources, targets, predictions)



#| label: fig-lstm-baseline-test-section-6-nags
#| fig-cap: Baseline LSTM - CDIP 243 - Test Section 6

plot_test_section(5, df_bins, sources, targets, predictions)





















#| label: fig-lstm-baseline-test-all
#| fig-cap: Baseline LSTM - All Test Sections
#| fig-subcap:
#|   - CDIP 225
#|   - CDIP 243
#| layout-nrow: 2

targets = pd.read_parquet("../testing_history/model_lstm.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.lstm_20241207_205411.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.lstm_20241207_205411.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.lstm_20241207_205411.parquet")
df_bins = pd.read_parquet("./model_input_spec_225.parquet")

targets *= 0.3
predictions *= 0.3


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.legend(labels=["Train", "Test"])
plt.show()

targets = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.lstm_20241207_215743.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.lstm_20241207_215743.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.lstm_20241207_215743.parquet")
df_bins = pd.read_parquet("./model_input_spec_243.parquet")

targets *= 0.3
predictions *= 0.3


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.legend(labels=["Train", "Test"])
plt.show()









#| label: fig-transformer-test-all
#| fig-cap: Transformer - All Test Sections
#| fig-subcap:
#|   - CDIP 225
#|   - CDIP 243
#| layout-nrow: 2

targets = pd.read_parquet("../testing_history/model_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.transformer_20241207_231033.parquet")
predictions = pd.read_parquet("../testing_history/model_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.transformer_20241207_231033.parquet")
sources = pd.read_parquet("../testing_history/model_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.transformer_20241207_231033.parquet")

df_bins = pd.read_parquet("./model_input_spec_225.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("Transformer", "225", targets, predictions))

targets = pd.read_parquet("../testing_history/model_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.transformer_20241207_235408.parquet")
predictions = pd.read_parquet("../testing_history/model_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.transformer_20241207_235408.parquet")
sources = pd.read_parquet("../testing_history/model_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.transformer_20241207_235408.parquet")

df_bins = pd.read_parquet("./model_input_spec_243.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()

result_stats.append(calc_stats("Transformer", "243", targets, predictions))









#| label: fig-transformer-lstm-100
#| fig-cap: LSTM 100 Epoch - All Test Sections
#| fig-subcap:
#|   - CDIP 225
#|   - CDIP 243
#| layout-nrow: 2

targets = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/test_results.targets.lstm_20241208_023701.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/test_results.predictions.lstm_20241208_023701.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/test_results.sources.lstm_20241208_023701.parquet")

df_bins = pd.read_parquet("./model_input_spec_225.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("100 Epoch LSTM", "225", targets, predictions))

targets = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/test_results.targets.lstm_20241208_061107.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/test_results.predictions.lstm_20241208_061107.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_100/test_results.sources.lstm_20241208_061107.parquet")

df_bins = pd.read_parquet("./model_input_spec_243.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("100 Epoch LSTM", "243", targets, predictions))










#| label: fig-transformer-lstm-4-layer
#| fig-cap: LSTM 4 Layer - All Test Sections
#| fig-subcap:
#|   - CDIP 225
#|   - CDIP 243
#| layout-nrow: 2

targets = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/test_results.targets.lstm_20241208_072749.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/test_results.predictions.lstm_20241208_072749.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/test_results.sources.lstm_20241208_072749.parquet")

df_bins = pd.read_parquet("./model_input_spec_225.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("4 Layer LSTM", "225", targets, predictions))

targets = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/test_results.targets.lstm_20241208_083208.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/test_results.predictions.lstm_20241208_083208.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_4.EPOCHS_25/test_results.sources.lstm_20241208_083208.parquet")

df_bins = pd.read_parquet("./model_input_spec_243.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("4 Layer LSTM", "243", targets, predictions))










#| label: fig-transformer-lstm-6-layer
#| fig-cap: LSTM 6 Layer - All Test Sections
#| fig-subcap:
#|   - CDIP 225
#|   - CDIP 243
#| layout-nrow: 2

targets = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/test_results.targets.lstm_20241208_093344.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/test_results.predictions.lstm_20241208_093344.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/test_results.sources.lstm_20241208_093344.parquet")

df_bins = pd.read_parquet("./model_input_spec_225.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()

result_stats.append(calc_stats("6 Layer LSTM", "225", targets, predictions))

targets = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/test_results.targets.lstm_20241208_110346.parquet")
predictions = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/test_results.predictions.lstm_20241208_110346.parquet")
sources = pd.read_parquet("../testing_history/model_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_6.EPOCHS_25/test_results.sources.lstm_20241208_110346.parquet")

df_bins = pd.read_parquet("./model_input_spec_243.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("6 Layer LSTM", "243", targets, predictions))









#| label: fig-enhc-transformer
#| fig-cap: Enhanced Transformer - All Test Sections
#| fig-subcap:
#|   - CDIP 225
#|   - CDIP 243
#| layout-nrow: 2

targets = pd.read_parquet("../testing_history/model_enhanced_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.enhanced_transformer_20241208_121422.parquet")
predictions = pd.read_parquet("../testing_history/model_enhanced_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.enhanced_transformer_20241208_121422.parquet")
sources = pd.read_parquet("../testing_history/model_enhanced_transformer.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.enhanced_transformer_20241208_121422.parquet")

df_bins = pd.read_parquet("./model_input_spec_225.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()

result_stats.append(calc_stats("Enhanced Transformer", "225", targets, predictions))

targets = pd.read_parquet("../testing_history/model_enhanced_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.enhanced_transformer_20241208_125950.parquet")
predictions = pd.read_parquet("../testing_history/model_enhanced_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.enhanced_transformer_20241208_125950.parquet")
sources = pd.read_parquet("../testing_history/model_enhanced_transformer.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.enhanced_transformer_20241208_125950.parquet")

df_bins = pd.read_parquet("./model_input_spec_243.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("Enhanced Transformer", "243", targets, predictions))









#| label: fig-enhc-lstm
#| fig-cap: Enhanced LSTM - All Test Sections
#| fig-subcap:
#|   - CDIP 225
#|   - CDIP 243
#| layout-nrow: 2

targets = pd.read_parquet("../testing_history/model_enhanced_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.enhanced_lstm_20241208_160252.parquet")
predictions = pd.read_parquet("../testing_history/model_enhanced_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.enhanced_lstm_20241208_160252.parquet")
sources = pd.read_parquet("../testing_history/model_enhanced_lstm.station_number_225.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.enhanced_lstm_20241208_160252.parquet")

df_bins = pd.read_parquet("./model_input_spec_225.parquet")


# start = 128
start = 0
# end = start + 128
end = -1

targets['vert_displacement_meters'].iloc[start:end].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].iloc[start:end].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()

result_stats.append(calc_stats("Enhanced LSTM", "225", targets, predictions))

targets = pd.read_parquet("../testing_history/model_enhanced_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.targets.enhanced_lstm_20241208_170202.parquet")
predictions = pd.read_parquet("../testing_history/model_enhanced_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.predictions.enhanced_lstm_20241208_170202.parquet")
sources = pd.read_parquet("../testing_history/model_enhanced_lstm.station_number_243.window_128.hidden_dim_128.NUM_LAYERS_2.EPOCHS_25/test_results.sources.enhanced_lstm_20241208_170202.parquet")

df_bins = pd.read_parquet("./model_input_spec_243.parquet")


targets['vert_displacement_meters'].plot(figsize=(16, 4), label="target", linewidth = 0.85)
predictions['vert_displacement_meters'].plot(label="prediction", linewidth=0.75)
plt.ylabel("Vertical Displacement [m]")
plt.show()
result_stats.append(calc_stats("Enhanced LSTM", "243", targets, predictions))

























results_df = pd.DataFrame(result_stats)





#| label: fig-results-mae-comparison
#| fig-cap: Mean Absolute Error Comparison by Model (Lower is better)

results_df = results_df.sort_values(["mae", "station"])

plt.figure(figsize=(6, 3.5))
sns.barplot(results_df, y="label", x="mae", hue="station")
for i in plt.gca().containers:
    plt.bar_label(i, fmt='%.2f', padding=3)
plt.ylabel(None);
plt.xlabel("Mean Absolute Error");





#| label: fig-results-mae-comparison-by-station
#| fig-cap: Mean Absolute Error Comparison by Station (Lower is better)

plt.figure(figsize=(8, 4.0))
sns.barplot(results_df, y="station", x="mae", hue="label")
for i in plt.gca().containers:
    plt.bar_label(i, fmt='%.2f', padding=3)
plt.ylabel(None);
plt.xlabel("Mean Absolute Error");
plt.legend(bbox_to_anchor=(1.25, 1.0),
          loc='upper center',
          ncol=1,
          frameon=False, title="Model")







#| label: fig-results-r2-comparison
#| fig-cap: $R^2$ Comparison by Model

results_df = results_df.sort_values(["r2"], ascending=False)

plt.figure(figsize=(6, 3.5))
sns.barplot(results_df, y="label", x="r2", hue="station")
for i in plt.gca().containers:
    plt.bar_label(i, fmt='%.2f', padding=3)
plt.ylabel(None);
plt.xlabel("$R^2$");





#| label: fig-results-r2-comparison-by-station
#| fig-cap: $R^2$ Comparison by Station

plt.figure(figsize=(8, 4.0))
sns.barplot(results_df, y="station", x="r2", hue="label")
for i in plt.gca().containers:
    plt.bar_label(i, fmt='%.2f', padding=3)
plt.ylabel(None);
plt.xlabel("$R^2$");
plt.legend(bbox_to_anchor=(1.25, 1.0),
          loc='upper center',
          ncol=1,
          frameon=False, title="Model")







#| label: fig-results-correlation-comparison
#| fig-cap: Pearson's Correlation Comparison by Model

results_df = results_df.sort_values(["correlation"], ascending=False)

plt.figure(figsize=(6, 3.5))
sns.barplot(results_df, y="label", x="correlation", hue="station")
for i in plt.gca().containers:
    plt.bar_label(i, fmt='%.2f', padding=3)
plt.ylabel(None);
plt.xlabel("Correlation");





#| label: fig-results-correlation-comparison-by-station
#| fig-cap: Pearson's Correlation Comparison by Station

plt.figure(figsize=(8, 4.0))
sns.barplot(results_df, y="station", x="correlation", hue="label")
for i in plt.gca().containers:
    plt.bar_label(i, fmt='%.2f', padding=3)
plt.ylabel(None);
plt.xlabel("Correlation");
plt.legend(bbox_to_anchor=(1.25, 1.0),
          loc='upper center',
          ncol=1,
          frameon=False, title="Model")





#| label: tbl-results-stats
#| tbl-cap: Results Summary

results_df
