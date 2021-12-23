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


from sklearn.inspection import permutation_importance
import logging
import os

class PFI:
    def __init__(self, base_path, n_samples=3000, n_repeats=10):
        self.base_path = base_path
        self.n_samples = n_samples
        self.n_repeats = n_repeats

    def compute_pfi(self, model_name):
        model = self.load_model(self.base_path, model_name)
        X, y, ids = self.load_data(model, self.n_samples)
        if X is None:
            raise ValueError("Unable to load data")
        fi = self.pfi(model, X, y, self.n_repeats)
        logging.info("PFI for %s completed" % model_name)
        return str(fi)

    @staticmethod
    def load_model(base_path, model_name):
        from .model import BaseModel
        print("Load model: %s..." % model_name)
        model_path = os.path.join(base_path, model_name, 'model.pth')
        model = BaseModel.load(model_path, torch.device('cpu'))
        model.eval()
        return model

    @staticmethod
    def load_data(model, n_samples=3000):
        from .classifier import BaseClassifier
        from .regressor import BaseRegressor
        from .vae import VAEandClassifier, VAEandRegressor
        from . import dataset as dt
        from . import utils

        dataset = model.config['dataset']
        if isinstance(model, VAEandClassifier):
            config = model.clf_model.config
            target_column_id = model.config['clf_target_column_id']
        elif isinstance(model, VAEandRegressor):
            config = model.regressor_model.config
            target_column_id = model.config['reg_target_column']
        else:
            config = model.config
            target_column_id = config['target_column_id']
        
        input_columns = config['input_columns']
        target_column = config['target_column']
        class_subset = config['class_subset'] if 'class_subset' in config else []
        types = config['types']
        columns = input_columns + [target_column_id]

        base_path = utils.get_project_path() / '.data' / dataset
        metadata_path = base_path / ('%s_meta.npy' % dataset)
        metadata = None
        if os.path.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True).item()
        
        device = torch.device('cpu')
        if isinstance(model, (BaseClassifier, VAEandClassifier)):
            train_data, valid_data, test_data, metadata, cw = dt.load_data_classification(
                dataset, model.transform, n_samples, int(1e7), False, device, 
                input_columns, types, target_column_id, True, class_subset)
        elif isinstance(model, (BaseRegressor, VAEandRegressor)):
            train_data, valid_data, test_data, metadata = dt.load_data_regression(
                dataset, model.transform, n_samples, int(1e7), False, device, 
                input_columns, types, target_column, True)
        X, y, ids = next(iter(test_data))
        return X, y, ids

    @staticmethod
    def pfi(model, X, y, n_repeats=10):
        from .vae import VAEandClassifier, VAEandRegressor
        if isinstance(model, VAEandClassifier):
            config = model.clf_model.config
        elif isinstance(model, VAEandRegressor):
            config = model.regressor_model.config
        else:
            config = model.config
        input_columns = config['input_columns']
        input_sizes = config['input_sizes']

        pi = permutation_importance(model, X, y, n_repeats=10, random_state=0)
        u = np.array([0] + input_sizes)
        s = {input_columns[i-1]: pi.importances_mean[u[:i].sum():u[:i+1].sum()].sum() for i in range(1, len(u))}
        return dict(sorted(s.items(), key=lambda item: item[1]))
