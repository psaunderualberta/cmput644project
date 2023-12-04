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
from dask.graph_manipulation import bind
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time
import dask

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dask.delayed
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


@dask.delayed(nout=2)
def delayed_execute_heuristic(df, heuristics, use_heuristic=True):
    if use_heuristic:
        new_cols = {str(h): h.execute(df) for h in heuristics}
        new_df = df.assign(**new_cols)
        x_cols = NORMALIZED_COLUMN_NAMES + list(new_cols.keys())
    else:
        new_df = df
        x_cols = NORMALIZED_COLUMN_NAMES
    return new_df[x_cols], new_df[CLASSES_2_Y_COLUMN]


@dask.delayed
def delayed_mean(df):
    return df.mean()


@dask.delayed
def delayed_variance(df):
    variances = df.var()
    variances[variances == 0] = 1
    return variances


@dask.delayed
def delayed_pooled_mean_and_var(means, variances, lens):
    means = np.array(means)
    variances = np.array(variances)
    lens = np.array(lens)[:, np.newaxis]
    return (
        np.sum(means * lens, axis=0) / np.sum(lens),
        np.sum((lens - 1) * variances, axis=0) / (np.sum(lens) - 1),
    )


@dask.delayed
def delayed_normalize(df, mu, std):
    return (df - mu) / std


@dask.delayed
def delayed_train(X, y, model):
    return model.fit(X, y)


@dask.delayed
def delayed_predict(X, model):
    return model.predict(X)


@dask.delayed
def delayed_get_scores(y_true, y_pred):
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    # Get accuracy, precision, recall, and f1 score
    return pd.Series(
        [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro"),
            recall_score(y_true, y_pred, average="macro"),
            f1_score(y_true, y_pred, average="macro"),
        ],
        index=["accuracy", "precision", "recall", "f1"],
    ).T


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
    heuristics = list(map(lambda h: parse_heuristic(h), heuristics))

    # Load parquet files
    dfs = []
    for file in COMBINED_DATA_FILES:
        dfs.append(delayed_load_data(file))

    Xs = [None for _ in dfs]
    ys = [None for _ in dfs]

    # Execute heuristics
    for i, df in enumerate(dfs):
        Xs[i], ys[i] = delayed_execute_heuristic(df, heuristics, use_heuristic=True)

    # Split data into train and test sets
    numfolds = 5
    splits = np.linspace(0, len(Xs), numfolds + 1, dtype=int)
    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    delayed_means_variances = []
    delayed_models = []
    delayed_scores = []

    for test_lower, test_upper in zip(splits, splits[1:]):
        X_trains.append(Xs[:test_lower] + Xs[test_upper:])
        y_trains.append(ys[:test_lower] + ys[test_upper:])
        X_tests.append(Xs[test_lower:test_upper])
        y_tests.append(ys[test_lower:test_upper])

        lens = [df.shape[0] for df in X_trains[-1]]
        means = [delayed_mean(df) for df in X_trains[-1]]
        variances = [delayed_variance(df) for df in X_trains[-1]]

        delayed_means_variances.append(
            delayed_pooled_mean_and_var(means, variances, lens)
        )

    start = time.time()
    means_variances = dask.compute(*delayed_means_variances)
    print("Time to compute means and variances: {}".format(time.time() - start))

    for X_train, y_train, (means, variances) in zip(
        X_trains, y_trains, means_variances
    ):
        model = dask.delayed(LogisticRegression)(warm_start=True)

        model_history = []
        for X, Y in zip(X_train, y_train):
            # Bind the current data to the 2nd previous model,
            # making sure that data loaded into memory is used relatively soon.
            # This is to prevent the memory from filling up and spilling to disk,
            # which can be in excess of 100GB when using heuristics.
            if len(model_history) > 2:
                X = bind(X, model_history[-2])

            X = delayed_normalize(X, means, variances**0.5)
            model = delayed_train(X, Y, model)
            model_history.append(model)

        delayed_models.append(model)

    start = time.time()
    models = dask.compute(*delayed_models)
    print("Time to train models: {}".format(time.time() - start))

    for X_test, y_test, model in zip(X_tests, y_tests, models):
        # Test model
        truths = []
        predictions = []
        for X, Y in zip(X_test, y_test):
            X = delayed_normalize(X, means, variances**0.5)
            truths.append(Y)
            predictions.append(delayed_predict(X, model))

        delayed_scores.append(delayed_get_scores(truths, predictions))

    start = time.time()
    scores = dask.compute(*delayed_scores)
    means = pd.concat(scores, axis=1).T.mean()
    # Save the scores
    fpath = os.path.join(os.path.dirname(artifact_path), "scores.csv")
    means.to_csv(fpath)
    print(means)
    print("Time to compute scores: {}".format(time.time() - start))



if __name__ == "__main__":
    main()
