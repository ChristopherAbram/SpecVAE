import numpy as np
import sys, os
import torch
import pandas as pd

import torchvision as tv
from torch.utils.data import DataLoader
from matchms import Spectrum

import specvae.utils as utils, specvae.dataset as dt
from specvae.dataset import MoNA
import specvae.vae as vae


def load_spectra_data(dataset, transform, n_samples=-1, device=None, cols=['spectrum']):
    test_data = None
    if dataset == 'MoNA':
        data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA.csv'
        df_train, df_valid, df_test = MoNA.get_by_split(data_path, columns=cols)
        print("Load test data")
        test_data = MoNA.preload_tensor(
            device=device, data_frame=df_test,
            transform=transform, limit=n_samples)
    return test_data


def spectrum2dense(spectrum: Spectrum, resolution=1., max_mz=2500, denormalize=True):
    mzs, ints = spectrum.peaks.mz, spectrum.peaks.intensities
    s = dt.spectrum_to_dense(list(zip(mzs,ints)), max_mz, resolution)
    # s = s * 100. if denormalize else s
    return s

def spectrum_processing(s):
    # s = normalize_intensities(s)
    return s

def parse_spectrum(row):
    string = row['spectrum']
    m = dt.SplitSpectrum()(string)
    mzs, ints = zip(*m)
    idx = np.argsort(np.array(mzs))
    s = Spectrum(mz=np.array(mzs)[idx], intensities=np.array(ints)[idx])
    # s = spectrum_processing(s)
    return s

class VAEScore:
    def __init__(self, model):
        self.model = model
        self.resolution = 1.#model.resolution
        self.max_mz = 2500 #model.max_mz
        self.device = model.device

    # def kl_divergence(self, x, y, relative=True, matrix=True):
    #     mu_x, logvar_x = self.model.encode(x)
    #     mu_y, logvar_y = self.model.encode(y)
    #     var_x, var_y = torch.exp(logvar_x), torch.exp(logvar_y)
    #     var_y_inv = 1. / var_y
    #     kl = 0.5 * torch.sum(logvar_y - logvar_x - 1 + 
    #         (var_y_inv * var_x) + (torch.square(mu_y - mu_x) * var_y_inv), dim=1)
    #     d = 1. / (1. + kl) if relative else kl
    #     return kl

    def _kl(self, mu_x, mu_y, logvar_x, logvar_y):
        var_x, var_y = torch.exp(logvar_x), torch.exp(logvar_y)
        var_y_inv = 1. / var_y
        kl = 0.5 * torch.sum(logvar_y - logvar_x - 1 + 
            (var_y_inv * var_x) + (torch.square(mu_y - mu_x) * var_y_inv), dim=1)
        return kl

    def spectrum2numpy(self, spectra, denormalize=False):
        if len(spectra) == 0:
            raise ValueError('List of spectra is empty')
        # Convert only when list contains elements of type Spectrum:
        if type(spectra[0]) is not Spectrum:
            return spectra
        s1 = spectrum2dense(spectra[0], self.resolution, self.max_mz, denormalize)
        s = np.zeros((len(spectra), s1.shape[0]), dtype=np.float32)
        s[0,:] = s1
        for i in range(1, len(spectra)):
            s[i,:] = spectrum2dense(spectra[i], self.resolution, self.max_mz, denormalize)
        return s
        
    def spectrum2tensor(self, spectra, denormalize=False, device=None):
        if not torch.is_tensor(spectra):
            spectra = torch.from_numpy(self.spectrum2numpy(spectra, denormalize))
        if device:
            spectra = spectra.to(device)
        return spectra

    def kl_divergence(self, x, y, relative=True, matrix=True, a=0.0005):
        # Convert list of matchms.Spectrum to pytorch tensor
        # TODO: optimize for symetric computation
        x = self.spectrum2tensor(x, device=self.model.device)
        y = self.spectrum2tensor(y, device=self.model.device)
        mu_x, logvar_x = self.model.encode(x)
        mu_y, logvar_y = self.model.encode(y)
        if matrix:
            xs, ys = x.shape[0], y.shape[0]
            kl = np.zeros((xs, ys))
            for i in range(xs):
                mu_xi = mu_x[i].unsqueeze(0).repeat(ys, 1)
                logvar_xi = logvar_x[i].unsqueeze(0).repeat(ys, 1)
                kl[i,:] = self._kl(mu_xi, mu_y, logvar_xi, logvar_y).data.cpu().numpy()
        else:
            kl = self._kl(mu_x, mu_y, logvar_x, logvar_y).data.cpu().numpy()
        kl = 1. / (1. + a*kl) if relative else kl
        # if relative:
        #     kl = -a * kl + 1
        #     kl[kl <= 0.] = 0.
        return kl

    def euclidean(self, x, y, relative=True, matrix=True, a=0.05):
        # Convert list of matchms.Spectrum to pytorch tensor
        x = self.spectrum2tensor(x, device=self.model.device)
        y = self.spectrum2tensor(y, device=self.model.device)
        mu_true, logvar_true = self.model.encode(x)
        mu_pred, logvar_pred = self.model.encode(y)
        d = torch.cdist(mu_true, mu_pred, p=2.0) if matrix else torch.norm(mu_true - mu_pred, dim=1)
        d = d.data.cpu().numpy()
        # if relative:
        #     d = -a * d + 1
        #     d[d <= 0.] = 0.
        d = 1. / (1. + a*d) if relative else d
        return d


def main(argc, argv):
    use_cuda = False
    cpu_device = torch.device('cpu')
    if torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda:0')
        print('GPU device count:', torch.cuda.device_count())
    else:
        device = torch.device('cpu')
    print('Device in use: ', device)

    
    dataset = 'MoNA' # HMDB and MoNA
    model_name = 'specvae_2500-500-50-500-2500 (20-06-2021_16-39-42)'
    spec_max_mz = 2500
    batch_size = 100

    cols = ['spectrum']

    # if matchms_spectrum:
    print("Load and preprocess %s validation data..." % dataset)
    if dataset == 'HMDB':
        valid_data_path = utils.get_project_path() / '.data' / 'HMDB' / 'hmdb_cfmid_dataset_valid.csv'
        df_valid = pd.read_csv(valid_data_path)

    elif dataset == 'MoNA':
        data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA.csv'
        df, df_valid, df_test = MoNA.get_by_split(data_path, columns=cols)

    X1 = df_valid.apply(parse_spectrum, axis=1).values

    # cols = ['spectrum', 'ionization mode', 'collision_energy_new']
    n_samples = 2 * batch_size # -1 if all
    spec_resolution = 1
    transform = tv.transforms.Compose([
        dt.SplitSpectrum(),
        dt.ToDenseSpectrum(resolution=spec_resolution, max_mz=spec_max_mz),
        # datasets.Ion2Int(one_hot=True)
    ])

    # Load and transform dataset:
    # test_data = load_spectra_data(dataset, transform, n_samples, device, cols)
    test_data = MoNA.preload_tensor(
        device=device, data_frame=df_valid,
        transform=transform, limit=n_samples)
    if test_data is None:
        print("No dataset specified, script terminates.")

    # Set data loaders:
    test_loader = DataLoader(
        dt.Spectra(data=test_data, device=device, columns=cols),
        batch_size=batch_size,
        shuffle=False
    )

    it = iter(test_loader)
    # spectrum_batch, mode_batch, energy_batch, id_batch = next(it)
    X2 = next(it)[0]
    y = next(it)[0]

    print("Load model: %s..." % model_name)
    model_path = utils.get_project_path() / '.model' / dataset / model_name / 'model.pth'
    model = vae.BaseVAE.load(model_path, device)
    model.eval()

    x1 = X1[:batch_size]
    x2 = X2[:batch_size]

    score = VAEScore(model)
    d1 = score.euclidean(x1, x1, relative=True)
    kl1 = score.kl_divergence(x1, x1, relative=True)

    d2 = score.euclidean(x2, x2, relative=True)
    kl2 = score.kl_divergence(x2, x2, relative=True)

    assert(np.allclose(d1, d2))
    assert(np.allclose(kl1, kl2))

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))