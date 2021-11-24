import numpy as np
import sklearn.metrics as skm

# from sklearn.metrics import \
#     accuracy_score, balanced_accuracy_score,\
#     recall_score, precision_score, f1_score

import torch
import torchmetrics as tm

def accuracy_score(y_true, y_pred):
    return tm.functional.accuracy(y_pred, y_true).item()

def balanced_accuracy_score(y_true, y_pred):
    n_classes = torch.unique(y_true).shape[0]
    return tm.functional.accuracy(y_pred, y_true, average='macro', num_classes=n_classes).item()

def recall_score(y_true, y_pred):
    return tm.functional.recall(y_pred, y_true).item()

def precision_score(y_true, y_pred):
    return tm.functional.precision(y_pred, y_true).item()

def f1_score(y_true, y_pred):
    return tm.functional.f1(y_pred, y_true).item()

def recall_score_macro(y_true, y_pred):
    # return skm.recall_score(y_true, y_pred, average='macro')
    n_classes = torch.unique(y_true).shape[0]
    return tm.functional.recall(y_pred, y_true, average='macro', num_classes=n_classes).item()

def precision_score_macro(y_true, y_pred):
    # return skm.precision_score(y_true, y_pred, average='macro')
    n_classes = torch.unique(y_true).shape[0]
    return tm.functional.precision(y_pred, y_true, average='macro', num_classes=n_classes).item()

def f1_score_macro(y_true, y_pred):
    # return skm.f1_score(y_true, y_pred, average='macro')
    n_classes = torch.unique(y_true).shape[0]
    return tm.functional.f1(y_pred, y_true, average='macro', num_classes=n_classes).item()

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import median_absolute_error as MedAE
from sklearn.metrics import explained_variance_score as explained_variance
from sklearn.metrics import max_error
from sklearn.metrics import r2_score as R2

def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred, squared=False)

def cosine_similarity(X, Y):
    cs = tm.functional.pairwise_cosine_similarity(X, Y)
    if X.shape[0] == Y.shape[0]:
        return (torch.trace(cs) / X.shape[0]).item()
    elif X.shape[0] == 1 or Y.shape[0] == 1:
        return cs.mean().item()
    else:
        raise ValueError('Unsupported input for cosine similarity')
    # cs = skm.pairwise.cosine_similarity(X, Y)
    # if X.shape[0] == Y.shape[0]: # one to one case
    #     return np.trace(cs) / X.shape[0]
    # elif X.shape[0] == 1 or Y.shape[0] == 1:
    #     return cs.mean()
    # else:
    #     raise ValueError('Unsupported input for cosine similarity')

def euclidean_distance(X, Y):
    # return np.mean(np.linalg.norm(X - Y, axis=1))
    return torch.mean(torch.norm(X - Y, dim=1)).item()

def mean_percentage_change(y_true, y_pred, eps=1e-7):
    # return np.mean((y_pred - y_true) / np.abs(y_true + eps))
    return torch.mean((y_pred - y_true) / torch.abs(y_true + eps)).item()

def mean_percentage_difference(y_true, y_pred, eps=1e-7):
    # return np.mean(np.abs(y_true - y_pred) / (y_true + y_pred + eps) / 2.)
    return torch.mean(torch.abs(y_true - y_pred) / (y_true + y_pred + eps) * 2.).item()

cos_sim = cosine_similarity
eu_dist = euclidean_distance
per_chag = mean_percentage_change
per_diff = mean_percentage_difference
