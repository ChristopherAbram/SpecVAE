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

def load_metadata(dataset):
    import os
    if dataset == 'MoNA':
        metadata_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA_meta.npy'
    elif dataset == 'HMDB':
        metadata_path = utils.get_project_path() / '.data' / 'HMDB' / 'HMDB_meta.npy'
    metadata = None
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
    return metadata


def main(argc, argv):
    # Processing parameters:
    session                 = 1
    model_name              = 'clf'
    dataset                 = 'MoNA' # HMDB and MoNA
    spec_max_mz             = 2500
    max_num_peaks_          = [10, 15, 25, 50]
    min_intensity_          = [0.01, 0.001, 0.0001, 0.1, 0.2, 0.5, 1.]
    rescale_intensity_      = [True, False]
    normalize_intensity     = True
    normalize_mass          = True
    n_samples               = -1 # -1 if all

    # Column settings:
    class_column_           = {
        'ionization_mode_id':       [],
        'instrument_id':            [0, 129, 130, 155, 161, 133, 122, 157, 136, 116, 115, 135],
        'instrument_type_id':       [1, 0, 7, 2, 10, 17, 12],
        'precursor_type_id':        [2, 3, 4, 1, 0, 28],
        'kingdom_id':               [],
        'superclass_id':            [6, 18, 14, 1, 10, 12, 0, 7, 5],
        'class_id':                 [198, 96, 32, 173, 61, 238, 94, 70, 106, 131, 65, 124, 43, 172, 190],
    }
    input_columns_          = [
        ['spectrum', 'collision_energy', 'total_exact_mass', 'precursor_mz',
        'ionization_mode_id', 'instrument_id', 'instrument_type_id', 'precursor_type_id', 
        'kingdom_id', 'superclass_id', 'class_id']
    ]

    # Train parameters:
    n_epochs                = 30
    batch_size              = 128
    learning_rate_          = [0.001]
    dropout_                = [0.0]
    enable_profiler         = False
    resume_training         = True

    # List of parameters:
    layers_config_ = [
        lambda indim: [indim, int(indim / 1.5), int(indim / 4)],
    ]

    # Load metadata:
    metadata = load_metadata(dataset)

    column_config = it.product(list(class_column_.items()), input_columns_)
    for (target_column, class_subset), input_columns in column_config:
        # Exclude target column from input columns:
        input_columns = input_columns.copy()
        input_columns.remove(target_column)
        types = [torch.float32] * len(input_columns)

        pre_configs = it.product(max_num_peaks_, min_intensity_, rescale_intensity_)
        for max_num_peaks, min_intensity, rescale_intensity in pre_configs:

            # Prepare transform:
            trans_list = [
                dt.SplitSpectrum(),
                dt.TopNPeaks(n=max_num_peaks),
                dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
                dt.Normalize(intensity=normalize_intensity, mass=normalize_mass, rescale_intensity=rescale_intensity),
                dt.ToMZIntConcatAlt(max_num_peaks=max_num_peaks),
            ]
            ohe_list = [dt.Int2OneHot(name, metadata[name]['n_class']) for name in input_columns if name in metadata]
            transform = tv.transforms.Compose(trans_list + ohe_list)

            all = False
            df = None
            if resume_training:
                try:
                    import pandas as pd
                    import os
                    filepath = utils.get_project_path() / '.model' / dataset / model_name / ('experiment%s.csv' % session)
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath, index_col=False, error_bad_lines=False)
                        all = True
                        for i, (layers_fn, learning_rate, dropout, beta) in enumerate(configs):
                            indim = 2 * max_num_peaks
                            layers = layers_fn(indim)
                            res = df[(df['param_max_num_peaks'] == max_num_peaks) & 
                                (df['param_min_intensity'] == min_intensity) & 
                                (df['param_rescale_intensity'] == rescale_intensity) & 
                                (df['param_normalize_intensity'] == normalize_intensity) & 
                                (df['param_normalize_mass'] == normalize_mass) & 
                                (df['param_max_mz'] == spec_max_mz) & 
                                (df['layer_config'].isin([str(layers)])) & 
                                (df['param_latent_dim'] == layers[0][-1]) & 
                                (df['param_beta'] == beta) & 
                                (df['param_n_samples'] == n_samples) & 
                                (df['param_n_epochs'] == n_epochs) & 
                                (df['param_batch_size'] == batch_size) & 
                                (df['param_learning_rate'] == learning_rate) & 
                                (df['param_dropout'] == dropout)]
                            if len(res) == 0:
                                all = False
                                break
                    else:
                        print("Session file doesn't exist")
                except Exception as e:
                    all = False
                    print("Error", e)

            if not all and resume_training:
                # Load and transform dataset:
                train_loader, valid_loader, test_loader, metadata, class_weights = dt.load_data_classification(
                    dataset, transform, n_samples, batch_size, True, device, 
                    input_columns, types, target_column, True, class_subset)

                configs = it.product(layers_config_, learning_rate_, dropout_)
                for i, (layers_fn, learning_rate, dropout) in enumerate(configs):
                    input_sizes = [2 * max_num_peaks]
                    input_sizes += [metadata[name]['n_class'] if name in metadata else 1 for name in input_columns if name != 'spectrum']
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
                        'target_column':        target_column.replace('_id', ''),
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
                        session=session,
                        metrics=['loss'],
                        evaluation_metrics=['accuracy_score', 'balanced_accuracy_score', 
                            'recall_score_macro', 'precision_score_macro', 'f1_score_macro'])
            else:
                print('Skip training for outer configuration...')
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
