import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from src.utility.scores import get_metrics
from sklearn.preprocessing import StandardScaler


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
    scores = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1"], dtype=np.float64)

    # Train and evaluate the learner on each fold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Get the train and test data
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Train the learner
        learner_copy = deepcopy(learner)
        learner_copy.fit(X_train, y_train)

        # Evaluate the learner using accuracy, fpr, fnr, and f1
        X_test = scaler.transform(X_test)
        y_pred = learner_copy.predict(X_test)
        stats = get_metrics(y_test, y_pred)

        # Append the scores to the list
        scores.loc[i] = [
            stats["acc"],
            stats["precision"],
            stats["recall"],
            stats["f1"],
        ]

    return scores.mean()
