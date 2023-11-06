import numpy as np
from .constants import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_metrics(actual, pred):
    # https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    # Accessed October 19th, 2023

    return {
        "precision": precision_score(actual, pred, average="macro"),
        "fpr": 1 - precision_score(actual, pred, average="macro"),
        "recall": recall_score(actual, pred, average="macro"),
        "fnr": 1 - recall_score(actual, pred, average="macro"),
        "acc": accuracy_score(actual, pred),
        "f1": f1_score(actual, pred, average="macro"),
    }
