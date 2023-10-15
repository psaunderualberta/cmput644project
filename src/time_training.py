from util.scores import fpr, fnr, f1_score, accuracy_score
from util.util import load_data

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import time
from util.constants import (
    SHORTENED_DATA_FILES,
    X_COLUMNS,
    CLASSES_2_Y_COLUMN,
)

if __name__ == "__main__":
    # Load the data
    t = time.time()
    df = load_data(SHORTENED_DATA_FILES)
    print("Time to load data: {:.2f}s".format(time.time() - t))

    # Scale the data using standard scaler
    t = time.time()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[X_COLUMNS])
    y = df[CLASSES_2_Y_COLUMN]
    print("Time to scale data: {:.2f}s".format(time.time() - t))

    # Train the model
    t = time.time()
    model = LogisticRegression(n_jobs=-1, random_state=42)
    model.fit(X, y)
    print("Time to train model: {:.2f}s".format(time.time() - t))

    # Evaluate the model
    t = time.time()
    y_pred = model.predict(X)
    print("Time to predict: {:.2f}s".format(time.time() - t))

    # Print accuracy, FPR, FNR, F1-score
    print("Accuracy: {:.2f}".format(accuracy_score(y, y_pred)))
    print("FPR: {:.2f}".format(fpr(y, y_pred)))
    print("FNR: {:.2f}".format(fnr(y, y_pred)))
    print("F1-score: {:.2f}".format(f1_score(y, y_pred)))
