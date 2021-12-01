import numpy as np

# import sklearn.metrics as skm
# from sklearn.metrics import \
#     accuracy_score, balanced_accuracy_score,\
#     recall_score, precision_score, f1_score
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.metrics import mean_squared_log_error as MSLE
# from sklearn.metrics import mean_absolute_error as MAE
# from sklearn.metrics import median_absolute_error as MedAE
# from sklearn.metrics import explained_variance_score as explained_variance
# from sklearn.metrics import max_error
# from sklearn.metrics import r2_score as R2
# def RMSE(y_true, y_pred):
#     return MSE(y_true, y_pred, squared=False)

import torch
import torchmetrics as tm

def accuracy_score(y_true, y_pred):
    return tm.functional.accuracy(y_pred, y_true)

def balanced_accuracy_score(y_true, y_pred):
    n_classes = torch.max(y_true).item() + 1
    return tm.functional.accuracy(y_pred, y_true, average='macro', num_classes=n_classes)

def recall_score(y_true, y_pred):
    return tm.functional.recall(y_pred, y_true)

def precision_score(y_true, y_pred):
    return tm.functional.precision(y_pred, y_true)

def f1_score(y_true, y_pred):
    return tm.functional.f1(y_pred, y_true)

def recall_score_macro(y_true, y_pred):
    # return skm.recall_score(y_true, y_pred, average='macro')
    n_classes = torch.max(y_true).item() + 1
    return tm.functional.recall(y_pred, y_true, average='macro', num_classes=n_classes)

def precision_score_macro(y_true, y_pred):
    # return skm.precision_score(y_true, y_pred, average='macro')
    n_classes = torch.max(y_true).item() + 1
    return tm.functional.precision(y_pred, y_true, average='macro', num_classes=n_classes)

def f1_score_macro(y_true, y_pred):
    # return skm.f1_score(y_true, y_pred, average='macro')
    n_classes = torch.max(y_true).item() + 1
    return tm.functional.f1(y_pred, y_true, average='macro', num_classes=n_classes)

def mean_squared_error(y_true, y_pred):
    return tm.functional.mean_squared_error(y_pred, y_true)

def root_mean_squared_error(y_true, y_pred):
    return tm.functional.mean_squared_error(y_pred, y_true, squared=False)

def mean_squared_log_error(y_true, y_pred):
    return tm.functional.mean_squared_log_error(y_pred, y_true)

def mean_absolute_error(y_true, y_pred):
    return tm.functional.mean_absolute_error(y_pred, y_true)

def explained_variance(y_true, y_pred):
    return tm.functional.explained_variance(y_pred, y_true)

def r2_score(y_true, y_pred):
    return tm.functional.r2_score(y_pred, y_true)

MSE = mean_squared_error
RMSE = root_mean_squared_error
MSLE = mean_squared_log_error
MAE = mean_absolute_error
R2 = r2_score


def cosine_similarity(X, Y):
    if X.shape[0] == Y.shape[0]:
        return torch.nan_to_num(tm.functional.cosine_similarity(Y, X, reduction='none')).mean()
    elif X.shape[0] == 1 or Y.shape[0] == 1:
        return torch.nan_to_num(tm.functional.pairwise_cosine_similarity(Y, X)).mean()
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
    return torch.mean(torch.nan_to_num(torch.norm(X - Y, dim=1)))

def mean_percentage_change(y_true, y_pred, eps=1e-7):
    # return np.mean((y_pred - y_true) / np.abs(y_true + eps))
    return torch.mean(torch.nan_to_num((y_pred - y_true) / torch.abs(y_true + eps)))

def mean_percentage_difference(y_true, y_pred, eps=1e-7):
    # return np.mean(np.abs(y_true - y_pred) / (y_true + y_pred + eps) / 2.)
    return torch.mean(torch.nan_to_num(torch.abs(y_true - y_pred) / (y_true + y_pred + eps) * 2.))

cos_sim = cosine_similarity
eu_dist = euclidean_distance
per_chag = mean_percentage_change
per_diff = mean_percentage_difference
