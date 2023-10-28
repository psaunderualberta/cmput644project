import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from scores import get_metrics


def cross_validation(df, data_classes, target_class, k, learner, random_state=42):
    """
    Perform k-fold cross validation on the given data and learner.
    """
    # Split the data into k folds
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    # Create the 'X' and 'y' data
    X = df[data_classes]
    y = df[target_class]

    # Create the results dataframe
    scores = pd.DataFrame(columns=["accuracy", "fpr", "fnr", "f1"], dtype=np.float64)

    # Train and evaluate the learner on each fold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Get the train and test data
        X_train = X.iloc[train_index]
        y_train = y[train_index]
        X_test = X.iloc[test_index]
        y_test = y[test_index]

        # Train the learner
        learner_copy = deepcopy(learner)
        learner_copy.fit(X_train, y_train)

        # Evaluate the learner using accuracy, fpr, fnr, and f1
        y_pred = learner_copy.predict(X_test)
        stats = get_metrics(y_test, y_pred)

        # Append the scores to the list
        scores.loc[i] = [
            stats["acc"],
            stats["fpr"],
            stats["fnr"],
            stats["f1"],
        ]

    return scores.mean()
