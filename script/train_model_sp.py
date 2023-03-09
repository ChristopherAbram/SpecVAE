from dataset import FilterPeaks, Normalize
import numpy as np
import sys, os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import itertools as it

import torchvision as tv
import train, utils, dataset as dt, visualize
from train import SpecVAETrainer
from dataset import MoNA
import vae

use_cuda = True
cpu_device = torch.device('cpu')
if torch.cuda.is_available() and use_cuda:
    device = torch.device('cuda:0')
    print('GPU device count:', torch.cuda.device_count())
else:
    device = torch.device('cpu')
print('Device in use: ', device)


def load_spectra_data(dataset, transform, n_samples=-1, device=None):
    train_data, valid_data, test_data = None, None, None
    if dataset == 'MoNA':
        data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA.csv'
        df_train, df_valid, df_test = MoNA.get_by_split(data_path, columns=['spectrum'])
        print("Load train data")
        train_data = MoNA.preload_tensor(
            device=device, data_frame=df_train, transform=transform, limit=n_samples)
        print("Load valid data")
        valid_data = MoNA.preload_tensor(
            device=device, data_frame=df_valid, transform=transform, limit=n_samples)
        print("Load test data")
        test_data = MoNA.preload_tensor(
            device=device, data_frame=df_test, transform=transform, limit=n_samples)
    return train_data, valid_data, test_data


def main(argc, argv):
    # Processing parameters:
    dataset = 'MoNA' # HMDB and MoNA
    spec_max_mz = 2500
    max_num_peaks = 1000
    min_intensity = 0.1
    n_samples = -1 # -1 if all

    # Train parameters:
    n_epochs = 50
    batch_size = 64

    # List of parameters:
    layers_configs = [
        #               Encoder,                      Decoder
        # lambda indim: [[indim, int(indim / 5), 200], [200, int(indim / 5), indim]],
        # lambda indim: [[indim, int(indim / 5), 150], [150, int(indim / 5), indim]],
        # lambda indim: [[indim, int(indim / 5), 200], [200, int(indim / 5), indim]],
        # lambda indim: [[indim, int(indim / 5), 100], [100, int(indim / 5), indim]],
        lambda indim: [[indim, int(indim / 1.3), 30], [30, int(indim / 1.3), indim]],
        # lambda indim: [[indim, int(indim / 1.3), int(indim / 5), 50], [50, int(indim / 5), int(indim / 1.3), indim]],
        # lambda indim: [[indim, int(indim / 1.3), 20], [20, int(indim / 1.3), indim]],
        # lambda indim: [[indim, int(indim / 1.3), 10], [10, int(indim / 1.3), indim]],
        # lambda indim: [[indim, int(indim / 5), 30], [30, int(indim / 5), indim]],
        # lambda indim: [[indim, int(indim / 5), 50], [50, int(indim / 5), indim]],
        # lambda indim: [[indim, int(indim / 5),  30], [30,  int(indim / 5), indim]],

        # lambda indim: [[indim, int(indim / 2), int(indim / 5), 50], [50, int(indim / 5), int(indim / 2), indim]],
        # lambda indim: [[indim, int(indim / 2), int(indim / 5), 30], [30, int(indim / 5), int(indim / 2), indim]],
        # lambda indim: [[indim, int(indim / 5), int(indim / 10), 150], [150, int(indim / 10), int(indim / 5), indim]],
        # lambda indim: [[indim, int(indim / 5), int(indim / 10), 100], [100, int(indim / 10), int(indim / 5), indim]],
        # lambda indim: [[indim, int(indim / 5), int(indim / 10),  50], [50,  int(indim / 10), int(indim / 5), indim]],
    ]

    learning_rates = [1e-03]
    configs = it.product(layers_configs, learning_rates)
    

    transform = tv.transforms.Compose([
        dt.SplitSpectrum(),
        dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
        dt.Normalize(intensity=True, mass=True, max_mz=spec_max_mz),
        dt.ToMZIntConcatAlt(max_num_peaks=max_num_peaks),
    ])
    model_transform = tv.transforms.Compose([
        dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
        dt.Normalize(intensity=True, mass=True, max_mz=spec_max_mz),
        dt.ToMZIntConcatAlt(max_num_peaks=max_num_peaks),
    ])

    revtv = tv.transforms.Compose([
        dt.ToMZIntDeConcatAlt(max_num_peaks=max_num_peaks),
        dt.Denormalize(intensity=True, mass=True, max_mz=spec_max_mz),
        dt.ToDenseSpectrum(resolution=0.05, max_mz=spec_max_mz)
    ])

    # Load and transform dataset:
    train_data, valid_data, test_data = load_spectra_data(dataset, transform, n_samples, device)
    if train_data is None:
        print("No dataset specified, abort!")
        return 1

    # Set data loaders:
    train_loader = DataLoader(
        dt.Spectra(data=train_data, device=device),
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dt.Spectra(data=valid_data, device=device),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dt.Spectra(data=test_data, device=device),
        batch_size=batch_size,
        shuffle=True
    )

    for i, (layers_fn, learning_rate) in enumerate(configs):
        indim = 2 * max_num_peaks
        layers = layers_fn(indim)

        print("Train model:")
        print("in_dim: ", indim)
        print("layers: ", layers)
        print("learning_rate: ", learning_rate)

        config = {
            'layer_config': layers,
            'transform': model_transform
        }

        # Create model
        model = vae.SpecAltVEA(config, device)
        trainer = train.SpecAltVAETrainer(model)
        trainer.compile(optimizer=optim.Adam(model.parameters(), lr=learning_rate))

        paths = train.prepare_training_session(trainer, dataset)

        # Train the model:
        history = trainer.fit(
            train_loader, epochs=n_epochs, 
            batch_size=batch_size, 
            validation_data=valid_loader,
            log_freq=100,
            visualization=lambda model, data_batch, dirpath, epoch: visualize.plot_spectra_grid(
                model, data_batch, dirpath, epoch, device, transform=revtv), 
            dirpath=paths['img_path'])

        train.export_training_session(trainer, paths, test_loader, n_samples)

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))