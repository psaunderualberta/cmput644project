import os
import pickle
import time
import glob
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from dask.graph_manipulation import bind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
from src.heuristic.parsing import parse_heuristic
from src.utility.constants import *


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


@dask.delayed(nout=2)
def delayed_trim_nans(X, y):
    X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y = y.reset_index(drop=True)
    idxs = np.where(pd.isnull(X).any(axis=1))[0]

    X = X.drop(idxs)
    y = y.drop(idxs)
    assert X.shape[0] == y.shape[0]
    return X, y


@dask.delayed
def delayed_mean(df):
    return df.mean()


@dask.delayed
def delayed_variance(df):
    variances = df.var(axis=0)
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


def train_model(pkl_path, USE_HEURISTIC=False):
    # Load the map-elites table
    with open(pkl_path, "rb") as f:
        tables = pickle.load(f)

    # Get unique heuristics
    heuristics, fitnesses = tables.get_stored_data(strip_nan=True, unique=True)
    heuristics = list(map(lambda h: parse_heuristic(h), heuristics))

    for h, f in zip(heuristics, fitnesses):
        print("Heuristic: {}, Fitness: {}".format(h, f))

    # Load parquet files
    dfs = []
    for file in SHORTENED_DATA_FILES:
        dfs.append(delayed_load_data(file))

    Xs = [None for _ in dfs]
    ys = [None for _ in dfs]

    # Execute heuristics
    for i, df in enumerate(dfs):
        Xs[i], ys[i] = delayed_execute_heuristic(
            df, heuristics, use_heuristic=USE_HEURISTIC
        )

        Xs[i], ys[i] = delayed_trim_nans(Xs[i], ys[i])


    # Split data into train and test sets
    numfolds = min(5, len(Xs) - 1)
    splits = np.linspace(0, len(Xs), numfolds + 1, dtype=int)
    X_trains = []
    y_trains = []
    X_tests = []
    y_tests = []
    delayed_means_variances = []
    delayed_models = []
    delayed_scores = []

    # Get datasets in each fold
    for test_lower, test_upper in zip(splits, splits[1:]):
        X_trains.append(Xs[:test_lower] + Xs[test_upper:])
        y_trains.append(ys[:test_lower] + ys[test_upper:])
        X_tests.append(Xs[test_lower:test_upper])
        y_tests.append(ys[test_lower:test_upper])

    # Get means and variances for each fold & the entire dataset
    for X_train in X_trains + [Xs]:
        lens = [df.shape[0] for df in X_train]
        means = [delayed_mean(df) for df in X_train]
        variances = [delayed_variance(df) for df in X_train]

        delayed_means_variances.append(
            delayed_pooled_mean_and_var(means, variances, lens)
        )

    start = time.time()
    means_variances = dask.compute(*delayed_means_variances)
    print("Time to compute means and variances: {}".format(time.time() - start))

    # Form the models as a delayed computation
    # NOTE: The last model is the model trained on the entire dataset
    for X_train, y_train, (means, variances) in zip(
        X_trains, y_trains, means_variances
    ):
        model = dask.delayed(LogisticRegression)(warm_start=True)

        # Bind the current data to the 2nd previous model,
        # making sure that data loaded into memory is used relatively soon.
        # This is to prevent the memory from filling up and spilling to disk,
        # which can be in excess of 100GB when using heuristics.
        model_history = []
        for X, Y in zip(X_train, y_train):
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

    # Make sure we have the correct number of scores
    assert len(delayed_scores) == numfolds

    start = time.time()
    scores = dask.compute(*delayed_scores)
    means = pd.concat(scores, axis=1).T.mean()

    # Save the scores
    if USE_HEURISTIC:
        folder = os.path.dirname(pkl_path)
    else:
        folder = os.path.join(Path(pkl_path).parents[1], "baseline")

    os.makedirs(folder, exist_ok=True)
    scores_fpath = os.path.join(folder, "scores.csv")
    means.to_csv(scores_fpath)
    print(means)
    print("Time to compute scores: {}".format(time.time() - start))

    model_path = os.path.join(folder, "models.pkl")
    data = {
        "models": models,
        "CV_score": means,
        "scores": scores,
        "variances": variances,
        "columns": dask.compute(Xs[0].columns),
        "heuristics": list(map(str, heuristics)),
        "fitnesses": fitnesses,
    }

    with open(model_path, "wb") as f:
        pickle.dump(data, f)


def main():
    versions = list(map(lambda v: f"v{v}", range(10, 11)))

    DOWNLOAD_ARTIFACTS = False
    if DOWNLOAD_ARTIFACTS:
        # Download all artifacts
        api = wandb.Api()
        for v in versions:
            try:
                artifact = api.artifact(f"psaunder/cmput644project/map-elites:{v}")
            except wandb.errors.CommError:
                print(f"Artifact {v} does not exist")
            else:
                artifact.download()

        # Replace all files called 'heuristic_storage.pkl' with 'tables.pkl'
        for src in glob(f"./artifacts/**/heuristic_storage.pkl", recursive=True):
            dest = src.replace("heuristic_storage.pkl", "tables.pkl")
            try:
                os.remove(dest)
            except FileNotFoundError:
                pass
            os.rename(src, dest)

    file_location = os.path.dirname(os.path.realpath(__file__))
    pkl_root = os.path.join(file_location, "artifacts")
    paths = ["local"]
    full_paths = [os.path.join(pkl_root, p, "tables.pkl") for p in paths]
    exist_paths = [p for p in full_paths if os.path.exists(p)]

    USE_HEURISTIC = True
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)
    print(client.dashboard_link)

    for p in exist_paths:
        client.restart()
        start = time.time()
        train_model(p, USE_HEURISTIC)
        print("Time to run whole pipeline: {}".format(time.time() - start))


if __name__ == "__main__":
    main()
