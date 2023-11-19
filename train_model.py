import os
import dask.dataframe as dd
import numpy as np
import wandb
import pickle
from src.utility.constants import *
from src.utility.util import load_data
import pandas as pd
from src.heuristic.parsing import parse_heuristic
from dask.distributed import Client, LocalCluster
from dask_ml.linear_model import LogisticRegression

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
    original_df = load_data(COMBINED_DATA_FILES, dask=True)

    new_cols = {str(h): h.execute(original_df) for h in heuristics}
    original_df = original_df.assign(**new_cols)

    x_columns = X_COLUMNS + list(map(str, heuristics))
    y_column = CLASSES_2_Y_COLUMN

    model = LogisticRegression(n_jobs=-1, random_state=42, solver="saga")

if __name__ == "__main__":
    main()