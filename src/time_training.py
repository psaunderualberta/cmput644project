from src.utility import load_data

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import time
from utility.constants import (
    COMBINED_DATA_FILES,
    SHORTENED_DATA_FILES,
    X_COLUMNS,
    CLASSES_2_Y_COLUMN,
)

from cross_validation import cross_validation

if __name__ == "__main__":
    # Load the data
    t = time.time()
    df = load_data(SHORTENED_DATA_FILES)
    df.reset_index(drop=True, inplace=True)
    print(df.head())
    print(df.nunique())
    print("Time to load data: {:.2f}s".format(time.time() - t))

    # Scale the data using standard scaler
    t = time.time()
    scaler = StandardScaler()
    df[X_COLUMNS] = scaler.fit_transform(df[X_COLUMNS])
    print("Time to scale data: {:.2f}s".format(time.time() - t))

    # Evaluate Logistic Regression using CV
    t = time.time()
    model = LogisticRegression(n_jobs=-1, random_state=42, solver="saga")
    results = cross_validation(df, X_COLUMNS, CLASSES_2_Y_COLUMN, 5, model)
    print("Time to run CV: {:.2f}s".format(time.time() - t))

    # Print results
    print(results)
