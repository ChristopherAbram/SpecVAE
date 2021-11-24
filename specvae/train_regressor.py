import sys
import torch
import numpy as np
import torchvision as tv
import itertools as it
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from specvae import train, utils
from specvae import dataset as dt
from specvae.regressor import BaseRegressor
from specvae.train import RegressorTrainer

use_cuda = True
device, cpu_device = utils.device(use_cuda, dev_name='cuda:0')


def main(argc, argv):
    # Processing parameters:
    model_name = 'reg'
    dataset = 'MoNA' # HMDB and MoNA
    spec_max_mz = 2500
    # max_num_peakss = [100, 500, 1000]
    # min_intensities = [0.1, 1.0, 5.0]
    max_num_peakss = [50]
    min_intensities = [0.1]
    normalize_intensity = True
    normalize_mass = True
    rescale_intensity = False
    n_samples = 2000 # -1 if all

    # Column settings
    target_column = 'total_exact_mass'
    input_columns = ['spectrum', 'collision_energy', 'instrument_type_id', 'precursor_type_id', 'kingdom_id', 'superclass_id']
    types = [torch.float32] * len(input_columns)

    enable_profiler = False

    # Train parameters:
    n_epochs = 5
    batch_size = 128

    # List of parameters:
    layers_configs = [
        lambda indim: [indim, int(indim / 2), int(indim / 4)],
    ]

    learning_rates = [1e-03]
    pre_configs = it.product(max_num_peakss, min_intensities)
    
    for max_num_peaks, min_intensity in pre_configs:
        configs = it.product(layers_configs, learning_rates)

        transform = tv.transforms.Compose([
            dt.SplitSpectrum(),
            dt.TopNPeaks(n=max_num_peaks),
            dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
            dt.Normalize(intensity=normalize_intensity, mass=normalize_mass, rescale_intensity=rescale_intensity),
            # dt.UpscaleIntensity(max_mz=spec_max_mz),
            dt.ToMZIntConcatAlt(max_num_peaks=max_num_peaks),
            # dt.Int2OneHot('instrument_id', 305),
            dt.Int2OneHot('instrument_type_id', 39),
            dt.Int2OneHot('precursor_type_id', 73),
            dt.Int2OneHot('kingdom_id', 2),
            dt.Int2OneHot('superclass_id', 19),
            # dt.Int2OneHot('class_id', 280),
            # dt.Int2OneHot('subclass_id', 405),
        ])

        # Load and transform dataset:
        train_loader, valid_loader, test_loader, metadata = dt.load_data_regression(
            dataset, transform, n_samples, batch_size, True, device, 
            input_columns, types, target_column, True)

        for i, (layers_fn, learning_rate) in enumerate(configs):
            input_sizes = [2 * max_num_peaks, 1, 39, 73, 2, 19]
            indim = np.array(input_sizes).sum()
            layers = layers_fn(indim)

            print("Train model:")
            print("layers: ", layers)
            print("learning_rate: ", learning_rate)

            config = {
                # Model params:
                'name':                 model_name,
                'layer_config':         np.array(layers),
                'dropout':              0.,
                'target_column':        target_column,
                'target_column_id':     target_column,
                'input_columns':        input_columns,
                'input_sizes':          input_sizes,
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
            model = BaseRegressor(config, device)
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
            
            trainer = RegressorTrainer(model, writer)
            trainer.compile(
                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                metrics=['loss'],
                evaluation_metrics=['RMSE', 'MAE'])

            # Train the model:
            history = trainer.fit(
                train_loader, epochs=n_epochs, 
                batch_size=batch_size, 
                validation_data=valid_loader, 
                log_freq=100, 
                visualization=None, 
                dirpath=paths['img_path'], 
                profiler=profiler)

            train.export_training_session(trainer, paths, 
                train_loader, valid_loader, test_loader, n_samples,
                metrics=['loss'],
                evaluation_metrics=['MSE', 'RMSE', 'MAE', 'R2', 'explained_variance'])

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
