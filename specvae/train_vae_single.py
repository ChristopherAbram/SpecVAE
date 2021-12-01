import sys
import os
import ast
import torch
import logging
import numpy as np
import torchvision as tv
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from specvae import train, utils, visualize
from specvae import dataset as dt
from specvae.vae import SpecVEA
from specvae.train import VAETrainer

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def boolean_string(s):
    if s not in {'False', 'false', 'True', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'


def main(argc, argv):
    # Set and parse arguments:
    parser = argparse.ArgumentParser(description='Train VAE model with metabolomics data')
    parser.add_argument('--session', type=str, help='Session number, used to identify model database with the process', default='01')
    parser.add_argument('--use-cuda', type=boolean_string, help='Train model on GPU if True, otherwise use CPU', default=True)
    parser.add_argument('--gpu-device', type=int, help='GPU device number', default=0)
    parser.add_argument('--model-name', type=str, help='A name of model', default='beta_vae')
    parser.add_argument('--dataset', type=str, choices=['MoNA', 'HMDB'], help='Name of a dataset used for training', default='MoNA')
    parser.add_argument('--n-samples', type=int, help='Number of samples used in training, -1 takes all available training samples', default=-1)
    parser.add_argument('--max-mz', type=float, help='Preprocessing parameter, maximum value for m/z parameter', default=2500)
    parser.add_argument('--n-peaks', type=int, help='Preprocessing parameter, maximum number n of top intensity peaks', default=50)
    parser.add_argument('--min-intensity', type=float, help='Preprocessing parameter, minimum intensity threshold', default=0.1)
    parser.add_argument('--rescale-intensity', type=boolean_string, help='Preprocessing parameter, normalize intensities to range min-max', default=False)
    parser.add_argument('--normalize-intensity', type=boolean_string, help='Preprocessing parameter, normalize intensities to range [0, 1]', default=True)
    parser.add_argument('--normalize-mass', type=boolean_string, help='Preprocessing parameter, normalize m/z values to range [0, 1]', default=True)
    parser.add_argument('--beta', type=float, help='Training parameter, beta parameter in beta-VAE', default=1.)
    parser.add_argument('--n-epochs', type=int, help='Training parameter, number of training epochs', default=20)
    parser.add_argument('--batch-size', type=int, help='Training parameter, batch size', default=128)
    parser.add_argument('--learning-rate', type=float, help='Training parameter, learning rate', default=0.001)
    parser.add_argument('--layer-config', type=str, 
        help='Model parameter, layer configuration for VAE, first layer and last layer has to be the same and equal to 2*n_peaks', 
        default='[[$indim, 15, 5],  [5, 15, $indim]]')
    parser.add_argument('--resume', type=boolean_string, help='Read associated session csv file and skip training if configuration already exists', default=True)
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

    # Column settings:
    input_columns           = ['spectrum']
    types                   = [torch.float32] * len(input_columns)

    # Train parameters:
    n_epochs                = args.n_epochs
    batch_size              = args.batch_size
    learning_rate           = args.learning_rate
    resume                  = args.resume
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

    # Set the transformation:
    transform = tv.transforms.Compose([
        dt.SplitSpectrum(),
        dt.TopNPeaks(n=max_num_peaks),
        dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
        dt.Normalize(intensity=normalize_intensity, mass=normalize_mass, rescale_intensity=rescale_intensity),
        dt.ToMZIntConcatAlt(max_num_peaks=max_num_peaks),
    ])

    revtrans = tv.transforms.Compose([
        dt.ToMZIntDeConcatAlt(max_num_peaks=max_num_peaks),
        dt.Denormalize(intensity=normalize_intensity, mass=normalize_mass, max_mz=spec_max_mz),
        dt.ToDenseSpectrum(resolution=0.05, max_mz=spec_max_mz)
    ])

    # Load and transform dataset:
    train_loader, valid_loader, test_loader, metadata = dt.load_data(
        dataset, transform, n_samples, batch_size, True, device, input_columns, types)

    config = {
        # Model params:
        'name':                 model_name,
        'layer_config':         np.array(layer_config),
        'latent_dim':           layer_config[0][-1],
        'beta':                 beta,
        'limit':                1.,
        'dropout':              0.,
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

    print("Train model:")
    for k, v in config.items():
        print(k, ':', str(v))

    try:
        # Create model:
        model = SpecVEA(config, device)
        paths = train.prepare_training_session(model, 
            subdirectory=config['dataset'], session_name=config['name'], session=session)

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
            optimizer=optim.Adam(model.parameters(), lr=config['learning_rate']),
            metrics=['loss', 'kldiv', 'recon'],
            evaluation_metrics=[])

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
            metrics=['loss', 'kldiv', 'recon'],
            evaluation_metrics=['cos_sim', 'eu_dist', 'per_chag', 'per_diff'])

    except Exception as e:
        logging.error("Error has occured while training the model: %s" % str(e))
        filepath = utils.get_project_path() / '.model' / config['dataset'] / config['name'] / ('error%s.csv' % session)
        hparams, rhparams = train.export_training_parameters(config, paths, session)
        if train.export_to_csv(hparams, rhparams, {}, paths, session, filepath):
            print("Add log to ", filepath)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
