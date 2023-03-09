import sys, os
import numpy as np
import torch
import itertools as it
import time

from specvae.model import BaseModel
from specvae import utils
from specvae import vae

from specvae.disentanglement import MoNA, HMDB
from specvae.disentanglement import compute_beta_vae
from specvae.disentanglement import compute_factor_vae
from specvae.disentanglement import compute_mig


class ModelLoader(object):
    def __init__(self, filepath, device=None):
        self._load_model(filepath, device)

    def _load_model(self, path, device=None):
        print("Load model: %s..." % path)
        model_path = os.path.join(path, 'model.pth')
        self.model = BaseModel.load(model_path, device)
        self.model.eval()

    def get_latent(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(self.model, vae.SpecVEA):
            X_, Z, latent_dist = self.model.forward_(X)
        return Z.cpu().detach().numpy()


def main(argc, argv):
    # Load model:
    device, cpu = utils.device(use_cuda=False)
    model_name = 'beta_vae_20-3-15-20_01 (06-12-2021_02-49-01)'
    path = utils.get_project_path() / '.model' / 'HMDB' / 'beta_vae' / model_name

    loader = ModelLoader(path, device)
    # Z = loader.get_latent(dset.spectrum)

    # Create ground thruth dataset:
    config = loader.model.config.copy()
    config['n_samples'] = 25000
    dset = HMDB(config)
    dset.load()

    # Metric config:
    batch_size_ = [128]
    num_train_ = [20000]
    latent_factor_indices_ = it.permutations([0, 1, 2], 2)
    # num_eval = 5000
    # num_variance_estimate = 5000

    options = it.product(batch_size_, num_train_, latent_factor_indices_)
    for batch_size, num_train, latent_factor_indices in options:

        dset.init_latent_factors(latent_factor_indices)

        num_eval = num_train if num_train < 10000 else int(num_train / 2)
        num_variance_estimate = num_train if num_train < 10000 else int(num_train / 2)

        print("Configuration:")
        print("latent_factor_indices", latent_factor_indices)
        print("batch_size", batch_size)
        print("num_train", num_train)
        print("num_eval", num_eval)
        print("num_variance_estimate", num_variance_estimate)

        start = time.time()
        beta_vae = compute_beta_vae(dset, loader.get_latent, 
            batch_size=batch_size, num_train=num_train, num_eval=num_eval)
        print(beta_vae)
        end = time.time()
        print("Time elapsed:", end - start)

        start = time.time()
        factor_vae = compute_factor_vae(dset, loader.get_latent, 
            batch_size=batch_size, num_train=num_train, num_eval=num_eval, 
            num_variance_estimate=num_variance_estimate)
        print(factor_vae)
        end = time.time()
        print("Time elapsed:", end - start)

        start = time.time()
        mig = compute_mig(dset, loader.get_latent, 
            batch_size=batch_size, num_train=num_train)
        print(mig)
        end = time.time()
        print("Time elapsed:", end - start)
    
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
