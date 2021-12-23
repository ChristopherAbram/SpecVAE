import os
from numpy.core.records import array
import pandas as pd
import numpy as np
import torch
from torch._C import dtype
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


def ion2int(mode):
    return 1 if mode.lower() == 'positive' else 0

class Identity:
    def __call__(self, x):
        return x

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

class FilterPeaks:
    def __init__(self, max_mz, min_intensity=0.):
        self.max_mz = max_mz
        self.min_intensity = min_intensity

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        spec = spec[np.where(spec[:,1] >= self.min_intensity)]
        spec = spec[np.where(spec[:,0] <= self.max_mz)]
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class TopNPeaks:
    def __init__(self, n):
        self.n = n

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        idx = np.argsort(spec[:,1])[::-1][:self.n]
        spec = spec[idx]
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec

class Normalize:
    def __init__(self, intensity=True, mass=True, rescale_intensity=False, min_intensity=0.1, max_mz=2500.):
        self.intensity = intensity
        self.mass = mass
        self.rescale_intensity = rescale_intensity
        self.min_intensity = min_intensity
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        if spec.shape[0] > 0:
            spec = np.array(spec)
            if self.rescale_intensity:
                mx, mn = spec[:,1].max(), spec[:,1].min()
                if not np.isclose(mx, mn):
                    mn = mn - self.min_intensity
                    spec[:,1] = (spec[:,1] - mn) * 100. / (mx - mn)
            if self.intensity:
                spec[:,1] = spec[:,1] * 0.01
            if self.mass:
                spec[:,0] = spec[:,0] * (1. / self.max_mz)
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec

class UpscaleIntensity:
    def __init__(self, max_mz=2500.):
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        spec[:,1] = spec[:,1] * self.max_mz / 100.
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class DeUpscaleIntensity:
    def __init__(self, max_mz=2500.):
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        spec[:,1] = spec[:,1] / self.max_mz * 100.
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec


class Denormalize:
    def __init__(self, intensity=True, mass=True, max_mz=2500.):
        self.intensity = intensity
        self.mass = mass
        self.max_mz = max_mz

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)
        if self.intensity:
            spec[:,1] = spec[:,1] * 100.
        if self.mass:
            spec[:,0] = spec[:,0] * self.max_mz
        if isinstance(sample, dict):
            sample['spectrum'] = spec
            return sample
        else:
            return spec

class ToMZIntConcat:
    def __init__(self, max_num_peaks, normalize=True):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)[:self.max_num_peaks]
        full = np.zeros((2 * self.max_num_peaks), dtype=np.float32)
        full[:spec.shape[0]] = spec[:,0]
        full[self.max_num_peaks:self.max_num_peaks + spec.shape[0]] = spec[:,1]
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

class ToMZIntConcatAlt:
    def __init__(self, max_num_peaks):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        spec = np.array(spec)[:self.max_num_peaks]
        full = np.zeros((2 * self.max_num_peaks), dtype=np.float32)
        idx = np.arange(0, 2 * self.max_num_peaks, 2)[:spec.shape[0]]
        full[idx] = spec[:,0]
        full[idx + 1] = spec[:,1]
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

class ToMZIntDeConcat:
    def __init__(self, max_num_peaks):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        mzs, ints = spec[:self.max_num_peaks], spec[self.max_num_peaks:]
        full = np.vstack((mzs, ints)).T
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

class ToMZIntDeConcatAlt:
    def __init__(self, max_num_peaks):
        self.max_num_peaks = max_num_peaks

    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        idx = np.arange(0, 2 * self.max_num_peaks, 2)
        mzs, ints = spec[idx], spec[idx + 1]
        # mzs, ints = spec[:self.max_num_peaks], spec[self.max_num_peaks:]
        full = np.vstack((mzs, ints)).T
        if isinstance(sample, dict):
            sample['spectrum'] = full
            return sample
        else:
            return full

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


class ToString:
    def __call__(self, sample):
        spec = sample['spectrum'] if isinstance(sample, dict) else sample
        ints = spec[:, 1].tolist()
        strs = ' '.join(['{:.4f}:{:.4f}'.format(mz, ints[i]) 
                            for i, mz in enumerate(spec[:, 0].tolist())])
        if isinstance(sample, dict):
            sample['spectrum'] = strs
            return sample
        else:
            return strs

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


class Ion2Int:
    def __init__(self, one_hot=True):
        self.one_hot = one_hot

    def __call__(self, sample):
        mode = sample['ionization mode'] if isinstance(sample, dict) else sample
        mode = ion2int(mode) if isinstance(mode, str) else mode
        mode = int(np.nan_to_num(mode))
        if self.one_hot:
            mode = np.eye(2, dtype=np.float32)[mode]
        else:
            mode = float(mode)
        if isinstance(sample, dict):
            sample['ionization mode'] = mode
            return sample
        else:
            return mode


class Int2OneHot:
    def __init__(self, col_name, n_classes=2):
        self.col_name = col_name
        self.n_classes = n_classes

    def __call__(self, sample):
        mode = sample[self.col_name] if isinstance(sample, dict) else sample
        mode = int(np.nan_to_num(mode))
        mode = np.eye(self.n_classes, dtype=np.float32)[mode]
        if isinstance(sample, dict):
            sample[self.col_name] = mode
            return sample
        else:
            return mode


class MergedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        d = self.dataset[index]
        return torch.cat(d[:-1]), d[-1]

class ClassificationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return torch.cat(data[:-2]), data[-2], data[-1]

class JointTrainingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        feat_ = torch.tensor([]) if len(data) <= 3 else torch.cat(data[1:-2])
        return data[0], feat_, data[-2], data[-1]

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
        data = []
        for col in self.columns:
            if self.device and col != 'id':
                self.data[col][index] = self.data[col][index].to(self.device)
            data.append(self.data[col][index] if col != 'id' else self.data[col][index].tolist())
        return tuple(data)

    @staticmethod
    def open(filepath, columns=None, split=False):
        if columns:
            if 'id' not in columns:
                columns = columns + ['id']
            if split:
                columns = columns + ['split']
            data_frame = pd.read_csv(filepath)[columns]
        else:
            data_frame = pd.read_csv(filepath)
        return data_frame

    @staticmethod
    def get_by_split(filepath=None, columns=None):
        df = Spectra.open(filepath, columns, split=True)
        train_df = df[df['split'].isin(['train'])]
        train_df = train_df.drop(['split'], axis=1)
        valid_df = df[df['split'].isin(['valid'])]
        valid_df = valid_df.drop(['split'], axis=1)
        test_df = df[df['split'].isin(['test'])]
        test_df = test_df.drop(['split'], axis=1)
        return train_df, valid_df, test_df

    @staticmethod
    def head(csv_file, n, dup=1, columns=None):
        df = Spectra.open(csv_file, columns)
        db = df.head(n)
        return db.loc[db.index.repeat(dup)]

    @staticmethod
    def get_unique(n, dup=1, columns=None, csv_file=None, df=None, random=False):
        if df is None:
            df = Spectra.open(csv_file, columns)
        ids = df['id'].unique()
        if n != 'all':
            if random:
                ids = ids[np.random.randint(low=0, high=len(ids), size=n)]
            else:
                ids = ids[:n]
        db = df[df['id'].isin(ids)] 
        return db.loc[db.index.repeat(dup)]

    @staticmethod
    def get_by_id(ids, dup=1, columns=None, csv_file=None, df=None):
        if df is None:
            df = MoNA.open(csv_file, columns)
        db = df[df['id'].isin(ids)] 
        return db.loc[db.index.repeat(dup)]

    @staticmethod
    def preload_tensor(device=None, filepath=None, columns=None, 
        transform=None, data_frame=None, limit=-1, types=None, do_print=True):

        columns = columns + ['id'] if columns and 'id' not in columns else None
        if data_frame is None:
            data_frame = Spectra.open(filepath, columns)
        
        if do_print:
            print("Load and transform...")
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
                if per % 5 == 0 and do_print:
                    print("Progress: {}%".format(per))
                per += 1
            i += 1
        
        if do_print:
            print("Convert data to pytorch tensors...")
        for i, col in enumerate(cols):
            if col == 'id':
                data[col] = np.array(data[col])
            if col == 'id' or col == 'InChIKey' or col == 'InChI':
                continue
            data[col] = torch.from_numpy(np.vstack((data[col])))
            data[col] = torch.where(torch.isnan(data[col]), torch.zeros_like(data[col]), data[col])
            if types is not None:
                data[col] = data[col].to(types[i])
            if device:
                data[col] = data[col].to(device)
        return data


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

class HMDB(Spectra):
    def __init__(self, filepath=None, columns=None, transform=None, data_frame=None, data=None):
        super(HMDB, self).__init__(transform=transform, data=data)
        self.columns = columns + ['HMDB'] if columns else None
        self.transform = transform
        if data:
            self.data = data

    @staticmethod
    def open(filepath, columns=None):
        if columns:
            columns = columns + ['HMDB']
            data_frame = pd.read_csv(filepath)[columns]
        else:
            data_frame = pd.read_csv(filepath, squeeze=False)
        data_frame['id'] = data_frame['HMDB']
        return data_frame


from . import utils
from torch.utils.data import DataLoader, WeightedRandomSampler

def load_metadata(dataset):
    import os
    metadata = None
    metadata_path = utils.get_project_path() / '.data' / dataset / ('%s_meta.npy' % dataset)
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
    return metadata

def load_spectra_data(dataset, transform, n_samples=-1, 
        device=None, columns=['spectrum'], types=[torch.float32], split=True, df=None, filename=None):
    train_data, valid_data, test_data = None, None, None
    if filename is None:
        data_path = utils.get_project_path() / '.data' / dataset / ('%s_full.csv' % dataset)
    else:
        data_path = utils.get_project_path() / '.data' / dataset / filename
    metadata = load_metadata(dataset)
    if split:
        df_train, df_valid, df_test = Spectra.get_by_split(data_path, columns=columns)
        print("Load train data")
        train_data = Spectra.preload_tensor(
            device=device, data_frame=df_train, transform=transform, limit=n_samples, types=types)
        print("Load valid data")
        valid_data = Spectra.preload_tensor(
            device=device, data_frame=df_valid, transform=transform, limit=n_samples, types=types)
        print("Load test data")
        test_data = Spectra.preload_tensor(
            device=device, data_frame=df_test, transform=transform, limit=n_samples, types=types)
        return train_data, valid_data, test_data, metadata
    else:
        if df is None:
            df = Spectra.open(data_path, columns=columns)
        print("Load data")
        data = Spectra.preload_tensor(
            device=device, data_frame=df, transform=transform, limit=n_samples, types=types, columns=columns)
        return data, metadata


def load_data(dataset, transform, n_samples=-1, batch_size=64, shuffle=True, 
        device=None, input_columns=['spectrum'], types=[torch.float32], split=True, df=None, filename=None):

    if split:
        train_data, valid_data, test_data, metadata = load_spectra_data(
            dataset, transform, n_samples, device, input_columns, types, True, filename=filename)

        if train_data is None:
            raise ValueError("No dataset specified, abort!")
        train_loader = DataLoader(
            MergedDataset(Spectra(data=train_data, device=device, columns=input_columns)), 
                batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(
            MergedDataset(Spectra(data=valid_data, device=device, columns=input_columns)), 
                batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(
            MergedDataset(Spectra(data=test_data, device=device, columns=input_columns)), 
                batch_size=batch_size, shuffle=shuffle)
        return train_loader, valid_loader, test_loader, metadata
    else:
        data, metadata = load_spectra_data(dataset, transform, n_samples, device, 
            input_columns, types, False, df, filename=filename)
        if data is None:
            raise ValueError("No dataset specified, abort!")
        loader = DataLoader(
            MergedDataset(Spectra(data=data, device=device, columns=input_columns)), 
                batch_size=batch_size, shuffle=shuffle)
        return loader, metadata



def load_spectra_data_classification(dataset, transform, n_samples=-1, device=None, 
        columns=['spectrum'], types=[torch.float32], 
        class_column=None, reject_noclass=True, class_subset=[]):
    train_data, valid_data, test_data = None, None, None
    data_path = utils.get_project_path() / '.data' / dataset / ('%s_full.csv' % dataset)
    metadata = load_metadata(dataset)
    df_train, df_valid, df_test = Spectra.get_by_split(data_path, columns=columns)
    if class_column is not None and reject_noclass:
        print("Reject samples with 'no-class' assigned")
        df_train = df_train[df_train[class_column] >= 0]
        df_valid = df_valid[df_valid[class_column] >= 0]
        df_test = df_test[df_test[class_column] >= 0]
    if class_column is not None and len(class_subset) > 0:
        print("Select classes from class_subset: ", class_subset)
        df_train = df_train.loc[df_train[class_column].isin(class_subset)]
        df_valid = df_valid.loc[df_valid[class_column].isin(class_subset)]
        df_test = df_test.loc[df_test[class_column].isin(class_subset)]
        print("Relabel classes ", class_subset, " to ", [i for i in range(len(class_subset))])
        df_train[class_column] = df_train.apply(lambda row: class_subset.index(row[class_column]), axis=1)
        df_valid[class_column] = df_valid.apply(lambda row: class_subset.index(row[class_column]), axis=1)
        df_test[class_column] = df_test.apply(lambda row: class_subset.index(row[class_column]), axis=1)
    
    types += [torch.long] # last type is used for label
    print("Load train data")
    train_data = Spectra.preload_tensor(
        device=device, data_frame=df_train, transform=transform, limit=n_samples, types=types)
    print("Load valid data")
    valid_data = Spectra.preload_tensor(
        device=device, data_frame=df_valid, transform=transform, limit=n_samples, types=types)
    print("Load test data")
    test_data = Spectra.preload_tensor(
        device=device, data_frame=df_test, transform=transform, limit=n_samples, types=types)
    return train_data, valid_data, test_data, metadata




def load_spectra_data_regression(dataset, transform, n_samples=-1, device=None, 
        columns=['spectrum'], types=[torch.float32], target_column=None, reject_novalue=True):
    
    train_data, valid_data, test_data = None, None, None
    data_path = utils.get_project_path() / '.data' / dataset / ('%s_full.csv' % dataset)
    metadata = load_metadata(dataset)
    df_train, df_valid, df_test = Spectra.get_by_split(data_path, columns=columns)
    if target_column is not None and reject_novalue:
        print("Reject samples with 'no-value' assigned")
        df_train.dropna(subset=[target_column], inplace=True)
        df_valid.dropna(subset=[target_column], inplace=True)
        df_test.dropna(subset=[target_column], inplace=True)
    
    types += [torch.float32] # last type is used for target
    print("Load train data")
    train_data = Spectra.preload_tensor(
        device=device, data_frame=df_train, transform=transform, limit=n_samples, types=types)
    print("Load valid data")
    valid_data = Spectra.preload_tensor(
        device=device, data_frame=df_valid, transform=transform, limit=n_samples, types=types)
    print("Load test data")
    test_data = Spectra.preload_tensor(
        device=device, data_frame=df_test, transform=transform, limit=n_samples, types=types)
    return train_data, valid_data, test_data, metadata



def create_weighted_sampler(data, metadata, class_column, device, class_subset=[]):
    n_classes = int(metadata[class_column]['n_class']) if len(class_subset) == 0 else len(class_subset)
    unique_class, class_count = torch.unique(data[class_column], return_counts=True, sorted=True)
    class_weights_ = 1. / torch.tensor(class_count, dtype=torch.float32)
    class_weights = torch.ones(n_classes, dtype=torch.float32).to(device)
    class_weights[unique_class] = class_weights_
    class_weights_sample = class_weights[data[class_column]].squeeze()
    if n_classes == 2 and unique_class.shape[0] == 2:
        class_weights = torch.tensor([class_count[0] / class_count[1]], dtype=torch.float32)
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_sample,
        num_samples=len(class_weights_sample),
        replacement=True)
    return weighted_sampler, class_weights


def load_data_classification(dataset, transform, n_samples=-1, batch_size=64, shuffle=True, 
        device=None, input_columns=['spectrum'], types=[torch.float32], 
        target_column=None, reject_noclass=True, class_subset=[], view=ClassificationDataset):
    class_weights = None
    columns = input_columns + [target_column]
    train_data, valid_data, test_data, metadata = load_spectra_data_classification(
        dataset, transform, n_samples, device, columns, types, target_column, reject_noclass, class_subset)
    if train_data is None:
        raise ValueError("No dataset specified, abort!")
    
    if target_column is not None:
        if not target_column in metadata:
            raise ValueError("Metadata for dataset '%s' doesn't contain information for target column '%s'" % (dataset, target_column))
        train_weighted_sampler, class_weights = create_weighted_sampler(train_data, metadata, target_column, device, class_subset)
        valid_weighted_sampler, valid_class_weights = create_weighted_sampler(valid_data, metadata, target_column, device, class_subset)
        test_weighted_sampler, test_class_weights = create_weighted_sampler(test_data, metadata, target_column, device, class_subset)

        train_loader = DataLoader(
            view(dataset=Spectra(data=train_data, device=device, columns=columns)),
            batch_size=batch_size, 
            sampler=train_weighted_sampler)
        valid_loader = DataLoader(
            view(dataset=Spectra(data=valid_data, device=device, columns=columns)),
            batch_size=batch_size, 
            sampler=valid_weighted_sampler)
        test_loader = DataLoader(
            view(dataset=Spectra(data=test_data, device=device, columns=columns)),
            batch_size=batch_size, 
            sampler=test_weighted_sampler)
    else:
        train_loader = DataLoader(
            Spectra(data=train_data, device=device, columns=columns), 
            batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(
            Spectra(data=valid_data, device=device, columns=columns), 
            batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(
            Spectra(data=test_data, device=device, columns=columns), 
            batch_size=batch_size, shuffle=shuffle)
    return train_loader, valid_loader, test_loader, metadata, class_weights





def load_data_regression(dataset, transform, n_samples=-1, batch_size=64, shuffle=True, 
        device=None, input_columns=['spectrum'], types=[torch.float32], 
        target_column=None, reject_novalue=True, view=ClassificationDataset):
    
    columns = input_columns + [target_column]
    train_data, valid_data, test_data, metadata = load_spectra_data_regression(
        dataset, transform, n_samples, device, columns, types, target_column, reject_novalue)
    if train_data is None:
        raise ValueError("No dataset specified, abort!")
    train_loader = DataLoader(
        view(dataset=Spectra(data=train_data, device=device, columns=columns)),
        batch_size=batch_size, 
        shuffle=shuffle)
    valid_loader = DataLoader(
        view(dataset=Spectra(data=valid_data, device=device, columns=columns)),
        batch_size=batch_size, 
        shuffle=shuffle)
    test_loader = DataLoader(
        view(dataset=Spectra(data=test_data, device=device, columns=columns)),
        batch_size=batch_size, 
        shuffle=shuffle)
    return train_loader, valid_loader, test_loader, metadata





def load_taxonomy_data(dataset, transform, n_samples=-1, device=None, cols=['spectrum'], field_name='class'):
    data, labels = None, None
    tax_fields = ('kingdom', 'superclass', 'class', 'subclass')
    field = tax_fields.index(field_name)
    def retrieve_filed(row):
        tax = row['taxonomy']
        arr = [f.split(':')[0] for f in tax.split(';')]
        row[field_name] = arr[field] if len(arr) > field else ''
        return row

    if dataset == 'HMDB':
        # HMDB:
        test_data_path = utils.get_project_path() / '.data' / 'HMDB' / 'hmdb_cfmid_dataset_test.csv'
        data_frame = HMDB.open(test_data_path, cols)
        df = data_frame
        df = MoNA.get_n_molecules(1000, df=df)
        df['HMDB'] = df['id']
        data_path = utils.get_project_path() / '.data' / 'HMDB' / 'HMDB_test_taxonomy.csv'
        tax_df = pd.read_csv(data_path)[['HMDB', 'taxonomy']]
        # data_frame = datasets.HMDB.open(data_path, cols)

        print("Get taxonomy by matching on HMDB (id) column...")
        tax_df = tax_df[tax_df['HMDB'].isin(df['id'].unique())]
        tax_df = tax_df.apply(retrieve_filed, axis=1)
        tax_df = tax_df.drop(['taxonomy'], axis=1)

        print("Derive distinct classes...")
        labels = np.unique(tax_df[field_name].to_numpy())
        labels = labels[labels != ''].tolist()
        print("Number of distinct %s: %d" % (field_name, len(labels)))

        print("Classify each row in dataset and assign label (index to class)...")
        df_merged = df.merge(tax_df, on='HMDB', suffixes=('','_y')).drop(['HMDB'], axis=1)
        def assign_label(row, labels):
            tax = row[field_name]
            try:
                row[field_name + '_id'] = labels.index(tax)
            except:
                row[field_name + '_id'] = -1
            return row
        df = df_merged.apply(lambda row: assign_label(row, labels), axis=1).drop([field_name], axis=1)

        print("Load test data")
        data = datasets.MoNA.preload_tensor(
            device=device, data_frame=df,
            transform=transform, limit=n_samples)

    elif dataset == 'MoNA':
        # MoNA:
        data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA.csv'
        df = MoNA.open(data_path, cols + ['InChI'])
        spectrum_str = df[['spectrum']].to_numpy()
        df = MoNA.get_unique(2000, df=df)
        tax_data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA_taxonomy.csv'
        tax_df = pd.read_csv(tax_data_path)[['InChI', 'taxonomy']]
        
        print("Get taxonomy by matching on inchi column...")
        tax_df = tax_df[tax_df['InChI'].isin(df['InChI'].unique())]
        tax_df = tax_df.apply(retrieve_filed, axis=1)
        tax_df = tax_df.drop(['taxonomy'], axis=1)

        print("Derive distinct classes...")
        labels = np.unique(tax_df[field_name].to_numpy())
        labels = labels[labels != ''].tolist()
        print("Number of distinct %s: %d" % (field_name, len(labels)))

        print("Classify each row in dataset and assign label (index to class)...")
        df_merged = df.merge(tax_df, on='InChI', suffixes=('','_y')).drop(['InChI'], axis=1)
        def assign_label(row, labels):
            tax = row[field_name]
            try:
                row[field_name + '_id'] = labels.index(tax)
            except:
                row[field_name + '_id'] = -1
            return row
        
        df = df_merged.apply(lambda row: assign_label(row, labels), axis=1).drop([field_name], axis=1)


        print("Load test data")
        data = MoNA.preload_tensor(
            device=device, data_frame=df,
            transform=transform, limit=n_samples)

    return data, labels