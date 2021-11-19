import numpy as np
import sklearn.metrics as skm

from sklearn.metrics import \
    accuracy_score, balanced_accuracy_score,\
    recall_score, precision_score, f1_score

def recall_score_macro(y_true, y_pred):
    return skm.recall_score(y_true, y_pred, average='macro')

def precision_score_macro(y_true, y_pred):
    return skm.precision_score(y_true, y_pred, average='macro')

def f1_score_macro(y_true, y_pred):
    return skm.f1_score(y_true, y_pred, average='macro')

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
    cs = skm.pairwise.cosine_similarity(X, Y)
    if X.shape[0] == Y.shape[0]: # one to one case
        return np.trace(cs) / X.shape[0]
    elif X.shape[0] == 1 or Y.shape[0] == 1:
        return cs.mean()
    else:
        raise ValueError('Unsupported input for cosine similarity')

def euclidean_distance(X, Y):
    return np.mean(np.linalg.norm(X - Y, axis=1))

def mean_percentage_change(y_true, y_pred, eps=1e-7):
    return np.mean((y_pred - y_true) / np.abs(y_true + eps))

def mean_percentage_difference(y_true, y_pred, eps=1e-7):
    return np.mean(np.abs(y_true - y_pred) / (y_true + y_pred + eps) / 2.)

cos_sim = cosine_similarity
eu_dist = euclidean_distance
per_chag = mean_percentage_change
per_diff = mean_percentage_difference
