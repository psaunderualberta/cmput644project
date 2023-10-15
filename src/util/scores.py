import numpy as np
from .constants import *

def fpr(actual, pred):
    # Convert to numpy arrays
    actual = np.array(actual)
    pred = np.array(pred)

    # Convert to 'attack' if anything other than 'benign'
    actual[actual != BENIGN_CLASS] = ATTACK_CLASS
    pred[pred != BENIGN_CLASS] = ATTACK_CLASS

    # Calculate the false positive rate
    # FPR = FP / (FP + TN)
    return np.mean((actual == BENIGN_CLASS) & (pred == ATTACK_CLASS))
        

def fnr(actual, pred):
    # Convert to numpy arrays
    actual = np.array(actual)
    pred = np.array(pred)

    # Convert to 'attack' if anything other than 'benign'
    actual[actual != BENIGN_CLASS] = ATTACK_CLASS
    pred[pred != BENIGN_CLASS] = ATTACK_CLASS

    # Calculate the false negative rate
    # FNR = FN / (FN + TP)
    return np.mean((actual == ATTACK_CLASS) & (pred == BENIGN_CLASS))

def f1_score(actual, pred):
    # Convert to numpy arrays
    actual = np.array(actual)
    pred = np.array(pred)

    # Convert to 'attack' if anything other than 'benign'
    actual[actual != BENIGN_CLASS] = ATTACK_CLASS
    pred[pred != BENIGN_CLASS] = ATTACK_CLASS

    # Calculate the F1-score
    # F1 = 2 * (precision * recall) / (precision + recall)
    precision = 1 - fpr(actual, pred)
    recall = 1 - fnr(actual, pred)
    return 2 * (precision * recall) / (precision + recall)

def accuracy_score(actual, pred):
    # Convert to numpy arrays
    actual = np.array(actual)
    pred = np.array(pred)

    # Calculate the accuracy
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    return np.mean(actual == pred)