from util.scores import fpr, fnr, f1_score, accuracy_score

import dask.dataframe as dd
import dask_ml.linear_model as dlm
import dask_ml.preprocessing as dpp
from dask.distributed import Client, LocalCluster

import time
from util.constants import COMBINED_DATA_FILE, X_COLUMNS, CLASSES_2_Y_COLUMN

if __name__ == "__main__":
    # cluster = LocalCluster()  # Launches a scheduler and workers locally
    # client = Client(cluster)
    # print(client.dashboard_link)

    # Load the data
    t = time.time()
    df = dd.read_csv(COMBINED_DATA_FILE)
    print("Time to load data: {:.2f}s".format(time.time() - t))

    # Scale the data using standard scaler
    t = time.time()
    scaler = dpp.StandardScaler()
    X = scaler.fit_transform(df[X_COLUMNS]).to_dask_array()
    y = df[CLASSES_2_Y_COLUMN].to_dask_array()
    print("Time to scale data: {:.2f}s".format(time.time() - t))

    # Train the model
    t = time.time()
    model = dlm.LogisticRegression()
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
