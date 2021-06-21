import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def spectrum_split_string(spec):
    return [[float(y) for y in x.split(':')] for x in spec.split(' ')]


def spectrum_to_dense(spec, max_mz, resolution):
    numbers = np.arange(0, max_mz, step=resolution, dtype=np.float32)
    result = np.zeros(len(numbers), dtype=np.float32)
    for i in spec:
        idx = np.searchsorted(numbers, i[0])
        try:
            result[idx] = i[1]
        except IndexError:
            result[-1] = i[1]
    return result


class SplitSpectrum:
    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        res = spectrum_split_string(spec)
        if isinstance(sample, dict):
            if 'id' in sample and sample['id'] != sample['id']:
                sample['id'] = '' 
            sample['spectrum'] = res
            return sample
        else:
            return res


class ToDenseSpectrum:
    def __init__(self, resolution, max_mz):
        self.resolution = resolution
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        result = spectrum_to_dense(spec, self.max_mz, self.resolution)
        if isinstance(sample, dict):
            # sample['spectrum'] = torch.from_numpy(result)
            sample['spectrum'] = result
            return sample
        else:
            return result

class ScaleSpectrum:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec *= self.scale
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class NormalizeSpectrum:
    def __init__(self, m=torch.Tensor([0.]), std=torch.Tensor([1.])):
        self.m = m
        self.std = std
        self.std += 1e-06 # avoid zeros

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = (spec - self.m) / self.std
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class ExpSpectrum:
    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.exp(spec) - 1.
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class DenormalizeSpectrum:
    def __init__(self, m=torch.Tensor([0.]), std=torch.Tensor([1.])):
        self.m = m
        self.std = std

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = spec * self.std + self.m
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class ToSparseSpectrum:
    def __init__(self, resolution, max_mz):
        self.resolution = resolution
        self.max_mz = max_mz

    def __call__(self, sample):
        spectrum = sample['spectrum'] if isinstance(sample, dict) else sample
        idx = (spectrum > 0).nonzero()[0]
        res = [(idx.astype(np.float32) * self.resolution).flatten(), spectrum[idx].flatten()]
        if isinstance(sample, dict):
            sample['spectrum'] = res
            return sample
        else:
            return res


class SparseToString:
    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        inte = spec[1].tolist()
        strs = ' '.join(['{:.4f}:{:.4f}'.format(mz, inte[i]) 
                            for i, mz in enumerate(spec[0].tolist())])
        if isinstance(sample, dict):
            sample['spectrum'] = strs
            return sample
        else:
            return strs



class Spectra(Dataset):
    def __init__(self, transform=None, data=None, columns=None, device=None):
        self.transform = transform
        self.data = data
        self.columns = ['spectrum'] if not columns else columns.copy()
        self.columns += ['id']
        self.device = device

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        data = []
        for col in self.columns:
            if self.device and col != 'id':
                self.data[col][index] = self.data[col][index].to(self.device)
            data.append(self.data[col][index])
        return tuple(data)


class MoNA(Spectra):
    def __init__(self, filepath=None, columns=None, transform=None, data_frame=None, data=None):
        super(MoNA, self).__init__(transform=transform, data=data)
        self.columns = columns + ['InChIKey'] if columns else None
        self.transform = transform
        if data:
            self.data = data
        else:
            self.data = MoNA.preload(filepath, columns, transform, data_frame)

    @staticmethod
    def get_by_split(filepath=None, columns=None, transform=None):
        df = MoNA.open(filepath, columns, split=True)
        train_df = df[df['split'].isin(['train'])]
        train_df = train_df.drop(['split'], axis=1)
        valid_df = df[df['split'].isin(['valid'])]
        valid_df = valid_df.drop(['split'], axis=1)
        test_df = df[df['split'].isin(['test'])]
        test_df = test_df.drop(['split'], axis=1)
        return train_df, valid_df, test_df

    # @staticmethod
    # def preload(filepath=None, columns=None, transform=None, data_frame=None):
    #     columns = columns + ['InChIKey'] if columns else None
    #     if data_frame is not None:
    #         data_frame = data_frame
    #     else:
    #         data_frame = MoNA.open(filepath, columns)
        
    #     data = []
    #     for index, sample in data_frame.iterrows():
    #         sample = sample.to_dict()
    #         if transform:
    #             sample = transform(sample)
    #         data.append(sample)
    #     return data

    @staticmethod
    def preload_tensor(device=None, filepath=None, columns=None, transform=None, data_frame=None, limit=-1):
        columns = columns + ['InChIKey'] if columns else None
        if data_frame is None:
            data_frame = MoNA.open(filepath, columns)
        
        print("Load and transform MoNA...")
        cols = data_frame.columns.tolist()
        data = {col: [] for col in cols}
        size = len(data_frame)
        size_per = size / 100
        i, per = 0, 1
        for index, sample in data_frame.iterrows():
            sample = sample.to_dict()
            if transform:
                sample = transform(sample)
            for col in cols:
                data[col].append(sample[col])
            if limit > 0 and i > limit:
                break
            if per * size_per <= i:
                if per % 5 == 0:
                    print("Progress: {}%".format(per))
                per += 1
            i += 1
        
        print("Convert data to pytorch tensors...")
        for col in cols:
            if col == 'id' or col == 'InChIKey':
                continue
            data[col] = torch.from_numpy(np.vstack((data[col])))
            data[col] = torch.where(torch.isnan(data[col]), torch.zeros_like(data[col]), data[col])
            if device:
                data[col] = data[col].to(device)
        return data

    @staticmethod
    def open(filepath, columns=None, split=False):
        if columns:
            columns = columns + ['InChIKey']
            if split:
                columns = columns + ['split']
            data_frame = pd.read_csv(filepath, index_col=0)[columns]
        else:
            data_frame = pd.read_csv(filepath, index_col=0)
        data_frame['id'] = data_frame['InChIKey']
        data_frame = data_frame.drop(['InChIKey'], axis=1)
        return data_frame

    @staticmethod
    def head(csv_file, n, dup=1, columns=None):
        df = MoNA.open(csv_file, columns)
        db = df.head(n)
        return db.loc[db.index.repeat(dup)]

    @staticmethod
    def get_unique(n, dup=1, columns=None, csv_file=None, df=None):
        if df is None:
            df = MoNA.open(csv_file, columns)
        ids = df['id'].unique()
        if n != 'all':
            ids = ids[:n]
        db = df[df['id'].isin(ids)] 
        return db.loc[db.index.repeat(dup)]

    @staticmethod
    def get_by_id(ids, dup=1, columns=None, csv_file=None, df=None):
        if df is None:
            df = MoNA.open(csv_file, columns)
        db = df[df['id'].isin(ids)] 
        return db.loc[db.index.repeat(dup)]
