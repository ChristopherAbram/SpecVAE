import sys
import torch
import numpy as np
import torchvision as tv
import itertools as it
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import train, utils, visualize
import dataset as dt
from vae import JointSpecGaussianVAEandRegressor
from train import JointVAEandRegressorTrainer

device, cpu_device = utils.device(use_cuda=True, dev_name='cuda:0')


def main(argc, argv):
    # Processing parameters:
    model_name              = 'specvae'
    dataset                 = 'MoNA' # HMDB and MoNA
    n_samples               = -1 # -1 if all
    spec_max_mz             = 2500
    max_num_peakss          = [50]
    min_intensities         = [0.1]
    normalize_intensity     = True
    normalize_mass          = True

    # Column settings:
    target_column           = 'collision_energy'
    input_columns           = ['spectrum']
    types                   = [torch.float32] * len(input_columns)
    class_subset            = []

    # Train parameters:
    n_epochs                = 50
    batch_size              = 128
    learning_rates          = [1e-03]
    dropouts                = [0.0]
    enable_profiler         = False

    # List of parameters:
    layers_configs          = [
        #               Encoder,         Decoder
        lambda indim: [[indim, 80, 25], [25, 80, indim]],
        lambda indim: [[indim, 80, 15], [15, 80, indim]],
        lambda indim: [[indim, 80, 10], [10, 80, indim]],
        lambda indim: [[indim, 80, 5],  [5,  80, indim]],
    ]
    sub_layers_configs      = [
        lambda indim: [indim, int(indim / 1.5)],
    ]

    pre_configs = it.product(max_num_peakss, min_intensities)
    for max_num_peaks, min_intensity in pre_configs:
        configs = it.product(layers_configs, sub_layers_configs, learning_rates, dropouts)

        transform = tv.transforms.Compose([
            dt.SplitSpectrum(),
            dt.TopNPeaks(n=max_num_peaks),
            dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
            # dt.Normalize(intensity=normalize_intensity, mass=normalize_mass, 
            #     rescale_intensity=False, max_mz=spec_max_mz, min_intensity=min_intensity),
            dt.UpscaleIntensity(max_mz=spec_max_mz),
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
            # dt.Denormalize(intensity=normalize_intensity, mass=normalize_mass, max_mz=spec_max_mz),
            dt.DeUpscaleIntensity(max_mz=spec_max_mz),
            dt.ToDenseSpectrum(resolution=0.05, max_mz=spec_max_mz)
        ])

        # Load and transform dataset:
        train_loader, valid_loader, test_loader, metadata = dt.load_data_regression(
            dataset, transform, n_samples, batch_size, True, device, 
            input_columns, types, target_column, True)

        for i, (layers_fn, sub_layers_fn, learning_rate, dropout) in enumerate(configs):
            indim = 2 * max_num_peaks
            layers = layers_fn(indim)
            sub_layers = sub_layers_fn(layers[0][-1])

            print("Train model:")
            print("layers: ", layers)
            print("learning_rate: ", learning_rate)

            config = {
                # 'name':                 model_name,
                'layer_config':         np.array(layers),
                'limit':                spec_max_mz,
                'dropout':              dropout,
                'transform':            transform,
                'input_columns':        input_columns,
                'types':                types,
                'regressor_config':     {
                    'layer_config':         sub_layers,
                    'dropout':              dropout,
                    'transform':            tv.transforms.Compose([dt.Identity()]),
                    'target_column':        target_column,
                    'input_columns':        input_columns,
                    'types':                types,
                    'dataset':              dataset,
                },
                'dataset':              dataset,
                'max_mz':               spec_max_mz,
                'min_intensity':        min_intensity,
                'max_num_peaks':        max_num_peaks,
                'normalize_intensity':  normalize_intensity,
                'normalize_mass':       normalize_mass,
                'n_samples':            n_samples,
            }

            # Create model:
            model = JointSpecGaussianVAEandRegressor(config, device)
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
            
            trainer = JointVAEandRegressorTrainer(model, writer)
            trainer.compile(
                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                metrics=['loss', 'kldiv', 'recon', 'reg_loss'],
                evaluation_metrics=['cos_sim'],
                evaluation_metrics_sub=['RMSE', 'MAE', 'max_error'])

            # Train the model:
            history = trainer.fit(
                train_loader, epochs=n_epochs, 
                batch_size=batch_size, 
                validation_data=valid_loader, 
                log_freq=10, 
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
