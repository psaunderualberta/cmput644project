import dask.dataframe as dd
from util.constants import (
    RAW_DATA_FILES,
    COMBINED_DATA_FILES,
    SHORTENED_DATA_FILES,
    CLASSES_8_MAPPING,
    CLASSES_2_MAPPING,
    CLASSES_8_Y_COLUMN,
    CLASSES_2_Y_COLUMN,
    CLASSES_34_Y_COLUMN,
)
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
    )
    print("Time to append columns: {:.2f}s".format(time.time() - t))


    # Sample 20 million rows, keeping the distribution of 2-classes
    n = 2e7
    t = time.time()
    frac = (n / df.shape[0]).compute()
    df_short = df.sample(frac=frac, random_state=42).reset_index(drop=True)
    print("Time to sample data: {:.2f}s".format(time.time() - t))

    # Output the shortened data to csv
    t = time.time()
    df_short.to_csv(SHORTENED_DATA_FILES)
    print("Time to output shortened data: {:.2f}s".format(time.time() - t))

    # Combine the data and output to csv
    # t = time.time()
    # df.to_csv(COMBINED_DATA_FILES, index=False)
    # print("Time to combine data: {:.2f}s".format(time.time() - t))
