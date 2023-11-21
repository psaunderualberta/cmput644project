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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cross_validation import cross_validation
import time
from dask import delayed

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@delayed
def delayed_load_data(file):
    df = pd.read_parquet(file)
    df.columns = list(
        map(
            lambda col: NORMALIZED_COLUMN_NAMES_MAPPING[col]
            if col in NORMALIZED_COLUMN_NAMES_MAPPING
            else col,
            df.columns,
        )
    )
    return df


@delayed(nout=2)
def delayed_execute_heuristic(df, heuristics, use_heuristic=True):
    if use_heuristic:
        new_cols = {str(h): h.execute(df) for h in heuristics}
        new_df = df.assign(**new_cols)
        x_cols = NORMALIZED_COLUMN_NAMES + list(new_cols.keys())
    else:
        new_df = df
        x_cols = NORMALIZED_COLUMN_NAMES
    return new_df[x_cols], new_df[CLASSES_2_Y_COLUMN]


@delayed
def delayed_mean(df):
    return df.mean()


@delayed
def delayed_variance(df):
    variances = df.var()
    variances[variances == 0] = 1
    return variances


@delayed
def delayed_pooled_mean_and_var(means, variances, lens):
    means = np.array(means)
    variances = np.array(variances)
    lens = np.array(lens)[:, np.newaxis]
    return (
        np.sum(means * lens, axis=0) / np.sum(lens),
        np.sum((lens - 1) * variances, axis=0) / (np.sum(lens) - 1),
    )


@delayed
def delayed_normalize(df, mu, std):
    return (df - mu) / std


@delayed
def delayed_train(X, y, model):
    return model.fit(X, y)


@delayed
def delayed_predict(X, model):
    return model.predict(X)


@delayed
def delayed_scores(y_true, y_pred):
    # Get accuracy, precision, recall, and f1 score
    return pd.Series(
        [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
        ],
        index=["accuracy", "precision", "recall", "f1"],
    )


@delayed
def delayed_mean_scores(scores):
    print(scores)
    return pd.concat(scores, axis=1).T.mean()


def main():
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    # Download the artifact from W&B
    api = wandb.Api()
    artifact = api.artifact("psaunder/cmput644project/map-elites:latest")
    artifact_path = os.path.join(artifact.download(), "tables.pkl")

    # Load the map-elites table
    with open(artifact_path, "rb") as f:
        tables = pickle.load(f)

    # Get unique heuristics
    heuristics, _ = tables.get_stored_data(strip_nan=True)
    heuristics = list(map(lambda h: parse_heuristic(h, dask=True), heuristics))

    # Load parquet files
    dfs = []
    for file in COMBINED_DATA_FILES:
        dfs.append(delayed_load_data(file))

    Xs = [None for _ in dfs]
    ys = [None for _ in dfs]

    # Execute heuristics
    for i, df in enumerate(dfs):
        Xs[i], ys[i] = delayed_execute_heuristic(df, heuristics, use_heuristic=False)

    split_index = int(len(Xs) * 0.8)
    X_train, y_train = Xs[:split_index], ys[:split_index]
    X_test, y_test = Xs[split_index:], ys[split_index:]

    lens = [df.shape[0] for df in X_train]
    means = [delayed_mean(df) for df in X_train]
    variances = [delayed_variance(df) for df in X_train]
    means, variances = delayed_pooled_mean_and_var(means, variances, lens).compute()

    # Train model
    model = LogisticRegression(warm_start=True)

    for X, Y in zip(X_train, y_train):
        X = delayed_normalize(X, means, variances**0.5)
        model = delayed_train(X, Y, model)

    model = model.compute()

    # Test model
    scores = []
    for X, Y in zip(X_test, y_test):
        X = delayed_normalize(X, means, variances**0.5)
        y_pred = delayed_predict(X, model)
        scores.append(delayed_scores(Y, y_pred))

    scores = delayed_mean_scores(scores).compute()
    print(scores)


if __name__ == "__main__":
    main()
