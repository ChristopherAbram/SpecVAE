import sys
import torch
import numpy as np
import torchvision as tv
import itertools as it
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import train, utils, visualize
import dataset as dt
from vae import SpecVEA, SpecGaussianVAE
from train import VAETrainer

device, cpu_device = utils.device(use_cuda=True, dev_name='cuda:0')


def main(argc, argv):
    # Processing parameters:
    model_name              = 'beta_vae'
    dataset                 = 'MoNA' # HMDB and MoNA
    n_samples               = -1 # -1 if all
    spec_max_mz             = 2500
    max_num_peaks_          = [10, 15, 25, 50]
    min_intensity_          = [0.1, 0.2, 0.5, 1.]
    # beta_                   = [0.01, 0.1, 0.2, 0.5, 1., 2., 5., 10., 100.]
    beta_                   = [1.]
    rescale_intensity_      = [True, False]
    normalize_intensity     = True
    normalize_mass          = True

    # Column settings:
    input_columns           = ['spectrum']
    types                   = [torch.float32] * len(input_columns)

    # Train parameters:
    n_epochs                = 20
    batch_size              = 128
    learning_rate_          = [0.001]
    dropout_                = [0.0]
    enable_profiler         = False

    # List of parameters:
    layers_configs = [
        #               Encoder,         Decoder
        lambda indim: [[indim, 15, 3],  [3, 15, indim]],
        lambda indim: [[indim, 15, 4],  [4, 15, indim]],
        lambda indim: [[indim, 15, 5],  [5, 15, indim]],
        lambda indim: [[indim, 15, 10], [10, 15, indim]],
    ]

    pre_configs = it.product(max_num_peaks_, min_intensity_, rescale_intensity_)
    for max_num_peaks, min_intensity, rescale_intensity in pre_configs:
        configs = it.product(layers_configs, learning_rate_, dropout_, beta_)

        transform = tv.transforms.Compose([
            dt.SplitSpectrum(),
            dt.TopNPeaks(n=max_num_peaks),
            dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
            dt.Normalize(intensity=normalize_intensity, mass=normalize_mass, rescale_intensity=rescale_intensity),
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


        for i, (layers_fn, learning_rate, dropout, beta) in enumerate(configs):
            indim = 2 * max_num_peaks
            layers = layers_fn(indim)

            print("Train model:")
            print("layers: ", layers)
            print("learning_rate: ", learning_rate)

            config = {
                # Model params:
                'name':                 model_name,
                'layer_config':         np.array(layers),
                'latent_dim':           layers[0][-1],
                'beta':                 beta,
                'limit':                1.,
                'dropout':              dropout,
                'input_columns':        input_columns,
                'types':                types,
                # Preprocessing params:
                'dataset':              dataset,
                'transform':            transform,
                'max_mz':               spec_max_mz,
                'min_intensity':        min_intensity,
                'max_num_peaks':        max_num_peaks,
                'normalize_intensity':  normalize_intensity,
                'normalize_mass':       normalize_mass,
                'rescale_intensity':    rescale_intensity,
                # Training parameters:
                'n_samples':            n_samples,
                'n_epochs':             n_epochs,
                'batch_size':           batch_size,
                'learning_rate':        learning_rate,
            }

            # Create model:
            model = SpecVEA(config, device)
            paths = train.prepare_training_session(model, subdirectory=dataset, session_name=model_name)

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
            
            trainer = VAETrainer(model, writer)
            trainer.compile(
                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                metrics=['loss', 'kldiv', 'recon'],
                evaluation_metrics=['cos_sim', 'eu_dist', 'per_chag', 'per_diff'])

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
                metrics=['loss', 'kldiv', 'recon'],
                evaluation_metrics=['cos_sim', 'eu_dist', 'per_chag', 'per_diff'])

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
