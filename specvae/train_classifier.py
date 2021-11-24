import sys
import torch
import numpy as np
import torchvision as tv
import itertools as it
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from specvae import train, utils
import specvae.dataset as dt
from specvae.classifier import BaseClassifier
from specvae.train import ClassifierTrainer

use_cuda = True
device, cpu_device = utils.device(use_cuda, dev_name='cuda:0')


def main(argc, argv):
    # Processing parameters:
    model_name = 'clf'
    dataset = 'MoNA' # HMDB and MoNA
    spec_max_mz = 2500
    # max_num_peakss = [100, 500, 1000]
    # min_intensities = [0.1, 1.0, 5.0]
    max_num_peakss = [50]
    min_intensities = [0.1]
    dropouts = [0.0]
    normalize_intensity = True
    normalize_mass = True
    rescale_intensity = False
    n_samples = 3000 # -1 if all

    # Column settings:
    class_column = 'ionization_mode'
    target_column = class_column + '_id'
    input_columns = ['spectrum', 'collision_energy', 'total_exact_mass', 'instrument_type_id', 'precursor_type_id', 'kingdom_id', 'superclass_id']
    types = [torch.float32] * len(input_columns)
    class_subset = []

    enable_profiler = False

    # Train parameters:
    n_epochs = 5
    batch_size = 128

    # List of parameters:
    layers_configs = [
        lambda indim: [indim, int(indim / 1.5), int(indim / 2.5), int(indim / 4)],
    ]

    learning_rates = [1e-03]
    pre_configs = it.product(max_num_peakss, min_intensities)
    
    for max_num_peaks, min_intensity in pre_configs:
        configs = it.product(layers_configs, learning_rates, dropouts)

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
        train_loader, valid_loader, test_loader, metadata, class_weights = dt.load_data_classification(
            dataset, transform, n_samples, batch_size, True, device, 
            input_columns, types, target_column, True, class_subset)

        for i, (layers_fn, learning_rate, dropout) in enumerate(configs):
            input_sizes = [2 * max_num_peaks, 1, 1, 39, 73, 2, 19]
            indim = np.array(input_sizes).sum()
            layers = layers_fn(indim)

            print("Train model:")
            print("layers: ", layers)
            print("learning_rate: ", learning_rate)

            config = {
                # Model params:
                'name':                 model_name,
                'layer_config':         np.array(layers),
                'n_classes':            metadata[target_column]['n_class'] if len(class_subset) == 0 else len(class_subset),
                'dropout':              dropout,
                'class_weights':        class_weights,
                'target_column':        class_column,
                'target_column_id':     target_column,
                'input_columns':        input_columns,
                'input_sizes':          input_sizes,
                'types':                types,
                # Preprocessing params:
                'dataset':              dataset,
                'transform':            transform,
                'class_subset':         class_subset,
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
            model = BaseClassifier(config, device)
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
            
            trainer = ClassifierTrainer(model, writer)
            trainer.compile(
                optimizer=optim.Adam(model.parameters(), lr=learning_rate),
                metrics=['loss'],
                evaluation_metrics=['accuracy_score', 'balanced_accuracy_score'])

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
                evaluation_metrics=['accuracy_score', 'balanced_accuracy_score', 
                    'recall_score_macro', 'precision_score_macro', 'f1_score_macro'])

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
