import dask.dataframe as dd
from src.utility.constants import (
    RAW_DATA_FILES,
    COMBINED_DATA_FILES,
    SHORTENED_DATA_FILES,
    CLASSES_8_MAPPING,
    CLASSES_2_MAPPING,
    CLASSES_8_Y_COLUMN,
    CLASSES_2_Y_COLUMN,
    CLASSES_34_Y_COLUMN,
    ATTACK_CLASS,
    ATTACK,
    BENIGN_CLASS,
    NORMALIZED_COLUMN_NAMES_MAPPING as mapping,
)
import numpy as np
import time
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    # Read in the data
    df = dd.read_csv(RAW_DATA_FILES)

    # Append the columns for the 8-class and 2-class task
    t = time.time()
    df[CLASSES_8_Y_COLUMN] = df[CLASSES_34_Y_COLUMN].apply(
        lambda x: CLASSES_8_MAPPING[x], meta=(CLASSES_8_Y_COLUMN, "object")
    )

    df[CLASSES_2_Y_COLUMN] = df[CLASSES_34_Y_COLUMN].apply(
        lambda x: CLASSES_2_MAPPING[x], meta=(CLASSES_2_Y_COLUMN, "int64")
    ).astype(np.int64)

    df = df.categorize(
        columns=[CLASSES_8_Y_COLUMN, CLASSES_2_Y_COLUMN, CLASSES_34_Y_COLUMN]
    )


    print("Time to append columns: {:.2f}s".format(time.time() - t))

    # Sample 20 million rows, keeping the distribution of 2-classes
    t = time.time()
    frac = 1 / 200  # Approximately 100MB
    df_short = df.sample(frac=frac, random_state=42).reset_index(drop=True)
    print("Time to sample data: {:.2f}s".format(time.time() - t))

    # Output the shortened data to csv
    t = time.time()
    df_short = df_short.reset_index(drop=True)
    df_short = df_short.repartition(partition_size="100MB")
    df_short.to_parquet("./shortened_data/", overwrite=True)
    print("Time to output shortened data: {:.2f}s".format(time.time() - t))

    # Combine the data and output to csv
    t = time.time()
    df = df.reset_index(drop=True)
    df = df.repartition(partition_size="100MB")
    df.to_parquet("./CICIoT2023/data/combined_data/", overwrite=True)
    print("Time to combine data: {:.2f}s".format(time.time() - t))
