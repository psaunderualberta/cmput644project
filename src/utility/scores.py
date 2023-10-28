import numpy as np
from .constants import *
from sklearn.metrics import confusion_matrix


def get_metrics(actual, pred):
    # https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    # Accessed October 19th, 2023
    cnf_matrix = confusion_matrix(actual, pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    return {
        "tpr": TP / (TP + FN),
        "fpr": FP / (FP + TN),
        "tnr": TN / (TN + FP),
        "fnr": FN / (FN + TP),
        "acc": (TP + TN) / (TP + FP + FN + TN),
        "f1": 2 * TP / (2 * TP + FP + FN),
    }
