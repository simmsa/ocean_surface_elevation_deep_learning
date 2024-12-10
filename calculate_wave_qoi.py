from pathlib import Path

import numpy as np
import pandas as pd

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


data_folders = [
    Path("./data/a2_std_partition/station_number=0225/").resolve(),
    Path("./data/a2_std_partition/station_number=0243/").resolve(),
]

for folder in data_folders:
    station_number = str(int(folder.parts[-1].split("=")[-1]))
    print("Finding all parquet files in", folder)
    data_files = folder.rglob("**/*.parquet")

    results = []

    for file in data_files:
        print("Working file: ", file)
        this_df = pd.read_parquet(file)

        qoi = calculate_wave_qoi(this_df, station_number, file)
        if qoi is not None:
            results.append(qoi)

    df = pd.DataFrame(results)
    df = df.set_index(["time"])
    df = df.sort_index()
    df.to_parquet(f"./data/b2_wave_qoi_stats/qoi_{station_number}.parquet")
