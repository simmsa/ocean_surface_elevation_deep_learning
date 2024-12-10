from pathlib import Path
import pandas as pd
from datetime import timedelta


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


if __name__ == "__main__":
    # Set output folder
    output_folder = Path("./data/a2_std_partition/").resolve()

    input_data = [
        {
            "station": 225,
            "path": Path("./data/a1_one_to_one_parquet/225_all.parquet").resolve(),
        },
        {
            "station": 243,
            "path": Path("./data/a1_one_to_one_parquet/243_all.parquet").resolve(),
        },
    ]

    # Process each DataFrame
    for input in input_data:
        df = pd.read_parquet(input["path"])
        station = input["station"]
        print(f"\nProcessing station {station}")
        print(f"Total samples: {len(df)}")
        partition_df(df, station, output_folder)
