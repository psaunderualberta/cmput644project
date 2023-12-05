import pandas as pd
import numpy as np
from dask_ml.model_selection import KFold
from copy import deepcopy
from src.utility.scores import get_metrics
from dask_ml.preprocessing import StandardScaler
import dask.dataframe as dd
import dask.array as da


def cross_validation(df, data_classes, target_class, k, learner, random_state=42):
    """
    Perform k-fold cross validation on the given data and learner.
    """
    # Split the data into k folds
    skf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Create the 'X' and 'y' data
    X = df[data_classes].to_dask_array(lengths=True)
    y = df[target_class].astype(np.int64).to_dask_array(lengths=True)

    # Create the results dataframe
    scores = dd.from_pandas(
        pd.DataFrame(
            columns=["accuracy", "precision", "recall", "f1"], dtype=np.float64
        ),
        npartitions=1,
    )

    # Train and evaluate the learner on each fold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Get the train and test data
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        # Remove columns that are all the same value
        constant_cols = da.all(X_train == X_train[0, :], axis=0).compute()
        X_train = X_train[:, ~constant_cols]
        X_test = X_test[:, ~constant_cols]

        # Compute the mean & std of all columns, then renormalize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        X_train = (X_train - mean) / std

        # Train the learner
        learner.fit(X_train, y_train)

        # Evaluate the learner using accuracy, fpr, fnr, and f1
        X_test = (X_test - mean) / std
        y_pred = learner.predict(X_test)

        # Compute the metrics, insert into scores
        scores = dd.concat([scores, get_metrics(y_test, y_pred)], axis=0)

    learner.reset()
    return scores.mean().compute()
