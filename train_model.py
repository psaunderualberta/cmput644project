import os
import dask.dataframe as dd
import numpy as np
import wandb
import pickle
from src.utility.constants import *
from src.utility.util import load_data
import pandas as pd
from src.heuristic.parsing import parse_heuristic
from dask.distributed import Client, LocalCluster, wait
from dask_ml.linear_model import LogisticRegression
from cross_validation import cross_validation
import time

def main():

    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    # Download the artifact from W&B
    api = wandb.Api()
    artifact = api.artifact('psaunder/cmput644project/map-elites:latest')
    artifact_path = os.path.join(artifact.download(), "tables.pkl")

    # Load the map-elites table
    with open(artifact_path, 'rb') as f:
        tables = pickle.load(f)
    
    # Get unique heuristics
    heuristics, _ = tables.get_stored_data(strip_nan=True)
    heuristics = list(np.unique(heuristics))
    heuristics = list(map(lambda h: parse_heuristic(h, dask=True), heuristics))

    # Load the data
    df = load_data(COMBINED_DATA_FILES, dask=True).repartition(npartitions=100)

    new_cols = {str(h): h.execute(df) for h in heuristics}
    df = df.assign(**new_cols)

    x_columns = NORMALIZED_COLUMN_NAMES + list(new_cols.keys())

    model = LogisticRegression(n_jobs=-1, random_state=42, solver="lbfgs")
    start = time.time()
    results = cross_validation(df, x_columns, CLASSES_2_Y_COLUMN, 5, model)
    print(results)
    print("Time to run CV: {:.2f}s".format(time.time() - start))

if __name__ == "__main__":
    main()