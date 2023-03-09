import sys
import torch
import numpy as np
import torchvision as tv
import torch.optim as optim
import argparse
import logging
import ast
from torch.utils.tensorboard import SummaryWriter

from specvae import train, utils, visualize
import specvae.dataset as dt
from specvae.utils import boolean_string
from specvae.vae import VAEandClassifier
from specvae.train import VAEandClassifierTrainer


def main(argc, argv):
    # Set and parse arguments:
    parser = argparse.ArgumentParser(description='Train VAE model jointly with classifier for metabolomics data')
    parser.add_argument('--session', type=str, help='Session number, used to identify model database with the process', default='01')
    parser.add_argument('--use-cuda', type=boolean_string, help='Train model on GPU if True, otherwise use CPU', default=False)
    parser.add_argument('--gpu-device', type=int, help='GPU device number', default=0)
    parser.add_argument('--model-name', type=str, help='A name of model', default='betavae_clf')
    parser.add_argument('--dataset', type=str, choices=['MoNA', 'HMDB'], help='Name of a dataset used for training', default='MoNA')
    parser.add_argument('--n-samples', type=int, help='Number of samples used in training, -1 takes all available training samples', default=-1)
    parser.add_argument('--max-mz', type=float, help='Preprocessing parameter, maximum value for m/z parameter', default=2500)
    parser.add_argument('--n-peaks', type=int, help='Preprocessing parameter, maximum number n of top intensity peaks', default=50)
    parser.add_argument('--min-intensity', type=float, help='Preprocessing parameter, minimum intensity threshold', default=0.001)
    parser.add_argument('--rescale-intensity', type=boolean_string, help='Preprocessing parameter, normalize intensities to range min-max', default=False)
    parser.add_argument('--normalize-intensity', type=boolean_string, help='Preprocessing parameter, normalize intensities to range [0, 1]', default=True)
    parser.add_argument('--normalize-mass', type=boolean_string, help='Preprocessing parameter, normalize m/z values to range [0, 1]', default=True)
    parser.add_argument('--beta', type=float, help='Training parameter, beta parameter in beta-VAE', default=1.0)
    parser.add_argument('--n-epochs', type=int, help='Training parameter, number of training epochs', default=1)
    parser.add_argument('--batch-size', type=int, help='Training parameter, batch size', default=128)
    parser.add_argument('--learning-rate', type=float, help='Training parameter, learning rate', default=0.001)
    parser.add_argument('--layer-config', type=str, 
        help='Model parameter, layer configuration for VAE, first layer and last layer has to be the same and equal to 2*n_peaks', 
        default='[[$indim, 15, 5],  [5, 15, $indim]]')
    parser.add_argument('--input-columns', type=str, help='Columns from dataset used as an input for classifier', 
        default='["spectrum", "collision_energy", "instrument_type_id"]')
    parser.add_argument('--target-column', type=str, help='Name of the target column for classification model', default='ionization_mode_id')
    parser.add_argument('--class-subset', type=str, 
        help='Subset of classes to use for classification, empty array means that all classes will participate in training', default='[]')
    parser.add_argument('--resume', type=boolean_string, 
        help='Read associated session csv file and skip training if configuration already exists', default=True)
    parser.add_argument('--preload', type=boolean_string, help='Whether to load and transform the entire dataset prior to training', default=False)
    args = parser.parse_args()

    # Processing and model parameters:
    use_cuda                = args.use_cuda
    gpu_device              = args.gpu_device
    session                 = args.session
    model_name              = args.model_name
    dataset                 = args.dataset # HMDB and MoNA
    n_samples               = args.n_samples # -1 if all
    spec_max_mz             = args.max_mz
    max_num_peaks           = args.n_peaks
    min_intensity           = args.min_intensity
    beta                    = args.beta
    rescale_intensity       = args.rescale_intensity
    normalize_intensity     = args.normalize_intensity
    normalize_mass          = args.normalize_mass
    layer_config            = ast.literal_eval(args.layer_config.replace('$indim', str(2*max_num_peaks)))

    # Classification model parameters:
    input_columns           = ast.literal_eval(args.input_columns)
    target_column           = args.target_column
    class_subset            = ast.literal_eval(args.class_subset)
    clf_layer_config        = lambda indim: [indim, int(indim / 1.5)]
    types                   = [torch.float32] * len(input_columns)

    # Train parameters:
    n_epochs                = args.n_epochs
    batch_size              = args.batch_size
    learning_rate           = args.learning_rate
    resume                  = args.resume
    preload                 = args.preload
    enable_profiler         = False

    # Open session log file:
    import os
    base_path = utils.get_project_path() / '.model' / dataset / model_name
    logfilepath = base_path / ('experiment%s.log' % session)
    os.makedirs(str(base_path), exist_ok=True)
    logging.basicConfig(
        filename=str(logfilepath), 
        format='[%(levelname)s] %(asctime)s (%(threadName)s): %(message)s', 
        level=logging.INFO)

    # Load metadata:
    metadata = dt.load_metadata(dataset)

    # Remove target_column from input_columns if one exists:
    input_columns = input_columns.copy()
    if target_column in input_columns:
        input_columns.remove(target_column)
    types = [torch.float32] * len(input_columns)

    # Compute input size and create relevant layers:
    input_sizes = [layer_config[0][-1]]
    input_sizes += [metadata[name]['n_class'] if name in metadata else 1 for name in input_columns if name != 'spectrum']
    indim = np.array(input_sizes).sum()
    clf_layers = clf_layer_config(indim)

    # Load experiment csv file and find current configuration:
    df = None
    skip_training = False
    if resume:
        try:
            import pandas as pd
            import os
            filepath = utils.get_project_path() / '.model' / dataset / model_name / ('experiment%s.csv' % session)
            if os.path.exists(filepath):
                logging.info("Found session file: %s" % str(filepath))
                df = pd.read_csv(filepath, index_col=False, error_bad_lines=False)
                # Find configuration:
                res = df[
                    (df['target_column_id'].isin(          [target_column])) & 
                    (df['input_columns'].isin(             [str(input_columns)])) & 
                    (df['class_subset'].isin(              [str(class_subset)])) & 
                    (df['param_max_num_peaks']          == max_num_peaks) & 
                    (df['param_min_intensity']          == min_intensity) & 
                    (df['param_rescale_intensity']      == rescale_intensity) & 
                    (df['param_normalize_intensity']    == normalize_intensity) & 
                    (df['param_normalize_mass']         == normalize_mass) & 
                    (df['param_max_mz']                 == spec_max_mz) & 
                    (df['layer_config'].isin(              [str(layer_config)])) & 
                    (df['param_latent_dim']             == layer_config[0][-1]) & 
                    (df['param_beta']                   == beta) & 
                    (df['param_n_samples']              == n_samples) & 
                    (df['param_n_epochs']               == n_epochs) & 
                    (df['param_batch_size']             == batch_size) & 
                    (df['param_learning_rate']          == learning_rate)]

                if len(res) > 0:
                    skip_training = True
            else:
                logging.info("Session file doesn't exist, run without resume")

        except Exception as e:
            # Don't fail process here
            logging.error(e, stack_info=True)

    if skip_training:
        logging.info("Model for this configuration has been trained already. SKIP %s" % str(args))
        return 0
    elif resume:
        logging.info("Configuration hasn't been trained yet. RUN")


     # Get device:
    device, cpu_device = utils.device(use_cuda=use_cuda, dev_name=('cuda:%d' % gpu_device))

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

    revtrans = tv.transforms.Compose([
        dt.ToMZIntDeConcatAlt(max_num_peaks=max_num_peaks),
        dt.Denormalize(intensity=normalize_intensity, mass=normalize_mass, max_mz=spec_max_mz),
        dt.ToDenseSpectrum(resolution=0.05, max_mz=spec_max_mz)
    ])

    # Load and transform dataset:
    train_loader, valid_loader, test_loader, metadata, class_weights = dt.load_data_classification(
        dataset, transform, n_samples, batch_size, True, device, 
        input_columns, types, target_column, True, class_subset, view=dt.JointTrainingDataset, preload=preload)

    config = {
        # Model params:
        'name':                 model_name,
        'layer_config':         np.array(layer_config),
        'latent_dim':           layer_config[0][-1],
        'beta':                 beta,
        'limit':                1.,
        'dropout':              0.,
        'input_columns':        ['spectrum'],
        'types':                [torch.float32],
        'clf_n_classes':        metadata[target_column]['n_class'] if len(class_subset) == 0 else len(class_subset),
        'clf_target_column':    target_column.replace('_id', ''),
        'clf_target_column_id': target_column,
        'clf_input_columns':    input_columns,
        'clf_input_sizes':      input_sizes,
        'clf_layer_config':     clf_layers,
        'clf_config':           {
            'layer_config':         clf_layers,
            'n_classes':            metadata[target_column]['n_class'] if len(class_subset) == 0 else len(class_subset),
            'dropout':              0.,
            'transform':            tv.transforms.Compose([dt.Identity()]),
            'class_weights':        class_weights,
            'target_column':        target_column.replace('_id', ''),
            'target_column_id':     target_column,
            'input_columns':        input_columns,
            'input_sizes':          input_sizes,
            'types':                types,
            'class_subset':         class_subset,
        },
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

    print("Train model:")
    for k, v in config.items():
        print(k, ':', str(v))

    try:
        # Create model:
        model = VAEandClassifier(config, device)
        paths = train.prepare_training_session(
            model, subdirectory=config['dataset'], session_name=config['name'], session=session)

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
        
        trainer = VAEandClassifierTrainer(model, writer)
        trainer.compile(
            optimizer=optim.Adam(model.parameters(), lr=config['learning_rate']),
            metrics=['loss', 'kldiv', 'recon'],
            evaluation_metrics=[],
            evaluation_metrics_sub=[])

        # Train the model:
        history = trainer.fit(
            train_loader, epochs=config['n_epochs'], 
            batch_size=config['batch_size'], 
            validation_data=valid_loader, 
            log_freq=10, 
            visualization=lambda model, data_batch, dirpath, epoch: visualize.plot_spectra_grid(
                model, data_batch, dirpath, epoch, device, transform=revtrans),
            dirpath=paths['img_path'], 
            profiler=profiler)

        train.export_training_session(trainer, paths, 
            train_loader, valid_loader, test_loader, config['n_samples'], 
            session=session,
            evaluation_metrics=['cos_sim', 'eu_dist', 'per_chag', 'per_diff'], 
            evaluation_metrics_sub=['accuracy_score', 'balanced_accuracy_score', 
                'recall_score_macro', 'precision_score_macro', 'f1_score_macro'])

    except Exception as e:
        logging.exception("Error has occured while training the model: %s" % str(e))
        filepath = utils.get_project_path() / '.model' / config['dataset'] / config['name'] / ('error%s.csv' % session)
        hparams, rhparams = train.export_training_parameters(config, paths, session)
        if train.export_to_csv(hparams, rhparams, {}, paths, session, filepath):
            print("Add log to ", filepath)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
