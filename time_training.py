import time

import numpy as np
from dask_ml.linear_model import LogisticRegression

from cross_validation import cross_validation
from src.utility.constants import (CLASSES_2_Y_COLUMN, CLASSES_8_Y_COLUMN,
                                   COMBINED_DATA_FILES, SHORTENED_DATA_FILES,
                                   X_COLUMNS)
from src.utility.util import load_data

if __name__ == "__main__":
    # Load the data
    t = time.time()
    df = load_data(COMBINED_DATA_FILES, dask=True)
    print(df.head())
    print("Time to load data: {:.2f}s".format(time.time() - t))

    # Evaluate Logistic Regression using CV
    t = time.time()
    model = LogisticRegression(n_jobs=-1, random_state=42)
    results = cross_validation(df, X_COLUMNS, CLASSES_2_Y_COLUMN, 5, model)
    print("Time to run CV: {:.2f}s".format(time.time() - t))

    # Print results
    print(results)
