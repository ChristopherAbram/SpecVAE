import sys
import torch
import numpy as np
import torchvision as tv
import itertools as it
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import train, utils, visualize
import dataset as dt
from jointvae import JointVAE, JointVAESigmoid
from train import JointVAETrainer

device, cpu_device = utils.device(use_cuda=True, dev_name='cuda:0')


def main(argc, argv):
    # Processing parameters:
    model_name              = 'specvae'
    dataset                 = 'MoNA' # HMDB and MoNA
    n_samples               = -1 # -1 if all
    spec_max_mz             = 2500
    max_num_peakss          = [15]
    # max_num_peakss          = [50]
    min_intensities         = [0.1]
    normalize_intensity     = True
    normalize_mass          = True

    # Column settings:
    input_columns           = ['spectrum']
    types                   = [torch.float32] * len(input_columns)

    # Train parameters:
    n_epochs                = 30
    batch_size              = 128
    learning_rates          = [1e-03]
    dropouts                = [0.0]
    enable_profiler         = False

    # List of parameters:
    layers_configs = [
        #               Encoder,         Decoder
        lambda indim: [[indim, 15, 3], [3, 15, indim]],
    ]
    latent_specs = [
        { 'cont': 3 }
    ]
    temperatures = [0.5] # only relevant for discrete variables
    cont_capacities = [
        # [0., 3., 10000, 5.],
        # [0., 3., 10000, 10.],
        # [0., 3., 10000, 20.],
        # [0., 3., 10000, 30.],
        # [0., 3., 10000, 40.],
        # [0., 3., 10000, 50.],
        # [0., 3., 10000, 100.],

        # [0., 5., 10000, 5.],
        # [0., 5., 10000, 10.],
        # [0., 5., 10000, 20.],
        # [0., 5., 10000, 30.],
        # [0., 5., 10000, 40.],
        # [0., 5., 10000, 50.],
        # [0., 5., 10000, 100.],

        # [0., 10., 10000, 5.],
        # [0., 10., 10000, 10.],
        [0., 10., 10000, 20.],
        [0., 10., 10000, 30.],
        [0., 10., 10000, 40.],
        [0., 10., 10000, 50.],
        [0., 10., 10000, 100.],

        [0., 20., 10000, 5.],
        [0., 20., 10000, 10.],
        [0., 20., 10000, 20.],
        [0., 20., 10000, 30.],
        [0., 20., 10000, 40.],
        [0., 20., 10000, 50.],
        [0., 20., 10000, 100.],

        [0., 50., 10000, 5.],
        [0., 50., 10000, 10.],
        [0., 50., 10000, 20.],
        [0., 50., 10000, 30.],
        [0., 50., 10000, 40.],
        [0., 50., 10000, 50.],
        [0., 50., 10000, 100.],

        # [0., 75., 10000, 10.],
        # [0., 75., 10000, 20.],
        # [0., 75., 10000, 30.],
        # [0., 75., 10000, 40.],
        # [0., 75., 10000, 50.],
        # [0., 75., 10000, 100.],
        # [0., 75., 10000, 500.],
        # [0., 75., 10000, 1000.],

        [0., 100., 10000, 5.],
        [0., 100., 10000, 10.],
        [0., 100., 10000, 20.],
        [0., 100., 10000, 30.],
        [0., 100., 10000, 40.],
        [0., 100., 10000, 50.],
        [0., 100., 10000, 100.],

        # [0., 150., 10000, 10.],
        # [0., 150., 10000, 20.],
        # [0., 150., 10000, 30.],
        # [0., 150., 10000, 40.],
        # [0., 150., 10000, 50.],
        # [0., 150., 10000, 100.],
        # [0., 150., 10000, 500.],
        # [0., 150., 10000, 1000.],

        # [0., 200., 10000, 10.],
        # [0., 200., 10000, 20.],
        # [0., 200., 10000, 30.],
        # [0., 200., 10000, 40.],
        # [0., 200., 10000, 50.],
        # [0., 200., 10000, 100.],
        # [0., 200., 10000, 500.],
        # [0., 200., 10000, 1000.],
    ]
    disc_capacities = [
        [0., 1., 10000, 10]
    ]

    pre_configs = it.product(max_num_peakss, min_intensities)
    for max_num_peaks, min_intensity in pre_configs:
        configs = it.product(layers_configs, latent_specs, temperatures, cont_capacities, disc_capacities, learning_rates, dropouts)

        transform = tv.transforms.Compose([
            dt.SplitSpectrum(),
            dt.TopNPeaks(n=max_num_peaks),
            dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
            dt.Normalize(intensity=normalize_intensity, mass=normalize_mass, rescale_intensity=True),
            #     rescale_intensity=False, max_mz=spec_max_mz, min_intensity=min_intensity),
            # dt.UpscaleIntensity(max_mz=spec_max_mz),
            dt.ToMZIntConcatAlt(max_num_peaks=max_num_peaks),
            # dt.Int2OneHot('instrument_type_id', 39),
            # dt.Int2OneHot('precursor_type_id', 73),
            # dt.Int2OneHot('superclass_id', 22),
            # dt.Int2OneHot('class_id', 253),
            # dt.Int2OneHot('subclass_id', 405),
            # dt.Int2OneHot('kingdom_id', 2),
        ])

        revtrans = tv.transforms.Compose([
            dt.ToMZIntDeConcatAlt(max_num_peaks=max_num_peaks),
            dt.Denormalize(intensity=normalize_intensity, mass=normalize_mass, max_mz=spec_max_mz),
            # dt.DeUpscaleIntensity(max_mz=spec_max_mz),
            dt.ToDenseSpectrum(resolution=0.05, max_mz=spec_max_mz)
        ])

        # Load and transform dataset:
        train_loader, valid_loader, test_loader, metadata = dt.load_data(
            dataset, transform, n_samples, batch_size, True, device, input_columns, types)

        for i, (layers_fn, latent_spec, temperature, cont_capacity, disc_capacity, learning_rate, dropout) in enumerate(configs):
            indim = 2 * max_num_peaks
            layers = layers_fn(indim)

            print("Train model:")
            print("layers: ", layers)
            print("learning_rate: ", learning_rate)

            config = {
                'layer_config':         np.array(layers),
                'limit':                spec_max_mz,
                'latent_spec':          latent_spec,
                'temperature':          temperature,
                'cont_capacity':        cont_capacity,
                'disc_capacity':        disc_capacity,
                'dropout':              dropout,
                'transform':            transform,
                'input_columns':        input_columns,
                'types':                types,
                'dataset':              dataset,
                'max_mz':               spec_max_mz,
                'min_intensity':        min_intensity,
                'max_num_peaks':        max_num_peaks,
                'normalize_intensity':  normalize_intensity,
                'normalize_mass':       normalize_mass,
                'n_samples':            n_samples,
            }

            # Create model:
            model = JointVAESigmoid(config, device)
            paths = train.prepare_training_session(model, dataset)

            profiler = None
            if enable_profiler:
                profiler = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(paths['training_path']),
                    record_shapes=True,
                    with_stack=True)

            writer = SummaryWriter(
                log_dir=paths['training_path'], 
                flush_secs=10)
            
            trainer = JointVAETrainer(model, writer)
            trainer.compile(
                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                metrics=['loss', 'kldiv', 'recon', 'kldiv_cont', 'cont_capacity_loss'],
                evaluation_metrics=['cos_sim'])

            # Train the model:
            history = trainer.fit(
                train_loader, epochs=n_epochs, 
                batch_size=batch_size, 
                validation_data=valid_loader, 
                log_freq=50, 
                visualization=lambda model, data_batch, dirpath, epoch: visualize.plot_spectra_grid(
                   model, data_batch, dirpath, epoch, device, transform=revtrans),
                dirpath=paths['img_path'], 
                profiler=profiler)

            train.export_training_session(trainer, paths, 
                train_loader, valid_loader, test_loader, n_samples, 
                metrics=['cos_sim'])

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
