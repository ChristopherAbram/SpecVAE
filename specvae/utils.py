import pathlib
import numpy as np
import sklearn.metrics.pairwise as smp

import matchms.similarity
from matchms import Spectrum
from matchms.filtering import add_precursor_mz
from matchms.filtering import normalize_intensities
from .dataset import ToSparseSpectrum
from matplotlib import pyplot as plt

import torch


def get_project_path():
    return pathlib.Path(__file__).parent.parent.absolute()

def get_attribute(config, name, default=None, required=True):
    if name in config:
        return config[name]
    elif required:
        raise ValueError('%s parameter missing' % name)
    return default


def cosine_similarity(x_true_batch, x_pred_batch):
    cs = smp.cosine_similarity(x_true_batch, x_pred_batch)
    return np.trace(cs) / x_true_batch.shape[0]

def boolean_string(s):
    if s not in {'False', 'false', 'True', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'


def device(use_cuda = True, dev_name='cuda:0'):
    cpu_device = torch.device('cpu')
    if torch.cuda.is_available() and use_cuda:
        device = torch.device(dev_name)
        print('GPU device count:', torch.cuda.device_count())
    else:
        device = torch.device('cpu')
    print('Device in use: ', device)
    return device, cpu_device


class ModifiedCosine:
    def __init__(self, resolution=0.05, max_mz=2500):
        self.mod_cos = matchms.similarity.ModifiedCosine()
        self.dense2sparse = ToSparseSpectrum(resolution, max_mz)

    def to_spectrum(self, x):
        x_ = self.dense2sparse(x)
        mzs, ints = x_[0].astype(np.float64), x_[1].astype(np.float64)
        pre = mzs.min() if len(mzs) > 0 else 0.01
        # pre = pre if pre > 0 else 0.01
        s = Spectrum(mz=mzs, intensities=ints, metadata={"precursor_mz": pre if pre > 0 else 0.01})
        s = add_precursor_mz(s)
        return s

    def compute(self, x_true, x_pred):
        try:
            ref = self.to_spectrum(x_true)
            que = self.to_spectrum(x_pred)
            score = self.mod_cos.pair(ref, que).tolist()[0]
            return score
        except:
            return 0.

    def __call__(self, x_true_batch, x_pred_batch):
        s_ = 0.
        for i in range(x_true_batch.shape[0]):
            s_ += self.compute(x_true_batch[i], x_pred_batch[i])
        return s_ / x_true_batch.shape[0]

def ignore_nan(arr):
    idx = np.where(np.isnan(arr)==False)
    return arr[idx]

def select_upper_triangular(arr, ignore_diagonal=True):
    d = 1 if ignore_diagonal else 0
    r, c = np.triu_indices(arr.shape[0], d) # with or without diagonal
    return arr[r, c]

def filter_score_matrix(data, is_symmetric=True, ignore_diagonal=True):
    if is_symmetric:
        data = select_upper_triangular(data, ignore_diagonal)
    data = ignore_nan(data.flatten())
    return data
