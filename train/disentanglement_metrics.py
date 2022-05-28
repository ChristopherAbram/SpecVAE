import sys, os
import argparse
from datetime import datetime
import logging
import time
import torch
import pandas as pd
import numpy as np
import multiprocessing as mp
import itertools as it

from specvae import utils
from specvae.model import BaseModel
from specvae import vae, jointvae

from specvae.disentanglement import MoNA, HMDB
from specvae.disentanglement import compute_beta_vae
from specvae.disentanglement import compute_factor_vae
from specvae.disentanglement import compute_mig


class DisentanglementMetric:
    def __init__(self, base_path, n_samples=15000, batch_size=128,
            n_train_samples=10000, n_eval_samples=5000, n_variance_estimate_samples=5000):
        self.base_path = base_path
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_train_samples = n_train_samples
        self.n_eval_samples = n_eval_samples
        self.n_variance_estimate_samples = n_variance_estimate_samples

    def compute(self, model_name):
        model = self.load_model(self.base_path, model_name)
        dset = self.get_ground_truth_data(model.config)
        dset.set_transform(model.config['transform'])
        dset.load()
        return self.compute_metrics(dset, model)

    def compute_metrics(self, dset, model):
        # Define all permutation of latent dimensions:
        dataset = model.config['dataset']
        if isinstance(model, vae.SpecVEA):
            latent_dim = model.config['latent_dim']
            latent_factor_indices_ = it.permutations(list(range(latent_dim)), latent_dim)
        elif isinstance(model, jointvae.JointVAE):
            latent_spec = model.config['latent_spec']
            cont_dim = latent_spec['cont']
            disc_dims = latent_spec['disc']
            latent_dim = int(cont_dim + len(disc_dims))
            latent_factor_indices_ = list(it.permutations(list(range(cont_dim)), cont_dim))
            latent_factor_indices_ = [tuple(list(lfi) + [cont_dim + i for i, d in enumerate(disc_dims)]) for lfi in latent_factor_indices_]
        else:
            raise RuntimeError('Unsupported model type')
        res = {
            'train': {'beta_vae': {}, 'factor_vae': {}, 'mig': {}}, 
            'eval':  {'beta_vae': {}, 'factor_vae': {}, 'mig': {}}
        }
        for latent_factor_indices in latent_factor_indices_:
            logging.info("Evaluate permutation: %s" % str(latent_factor_indices))
            dset.init_latent_factors(latent_factor_indices)
            # BetaVAE score:
            start = time.time()
            beta_vae = compute_beta_vae(dset, lambda X: self.get_latent_representation(model, X), 
                batch_size=self.batch_size, num_train=self.n_train_samples, num_eval=self.n_eval_samples)
            end = time.time()
            logging.info("beta_vae: time elapsed: %s" % (end - start))
            # FactorVAE score:
            start = time.time()
            factor_vae = compute_factor_vae(dset, lambda X: self.get_latent_representation(model, X), 
                batch_size=self.batch_size, num_train=self.n_train_samples, num_eval=self.n_eval_samples, 
                num_variance_estimate=self.n_variance_estimate_samples)
            end = time.time()
            logging.info("factor_vae: time elapsed: %s" % (end - start))
            # MIG score:
            start = time.time()
            mig = compute_mig(dset, lambda X: self.get_latent_representation(model, X), 
                batch_size=self.batch_size, num_train=self.n_train_samples)
            end = time.time()
            logging.info("mig: time elapsed: %s" % (end - start))
            # Extract values:
            res['train']['beta_vae'][latent_factor_indices]     = beta_vae['train_accuracy']
            res['eval']['beta_vae'][latent_factor_indices]      = beta_vae['eval_accuracy']
            res['train']['factor_vae'][latent_factor_indices]   = factor_vae['train_accuracy']
            res['eval']['factor_vae'][latent_factor_indices]    = factor_vae['eval_accuracy']
            res['train']['mig'][latent_factor_indices]          = mig['discrete_mig']
            res['eval']['mig'][latent_factor_indices]           = mig['discrete_mig']
        return res

    @staticmethod
    def load_model(base_path, model_name):
        print("Load model: %s..." % model_name)
        model_path = os.path.join(base_path, model_name, 'model.pth')
        model = BaseModel.load(model_path, torch.device('cpu'))
        model.eval()
        return model

    def get_ground_truth_data(self, config):
        if 'dataset' not in config:
            raise RuntimeError('Model\'s config doesn\'t contain dataset name')
        config = config.copy()
        config['n_samples'] = self.n_samples
        dset = None
        if config['dataset'] == 'MoNA':
            dset = MoNA(config)
        elif config['dataset'] == 'HMDB':
            dset = HMDB(config)
        return dset

    def get_latent_representation(self, model, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(model, vae.SpecVEA):
            with torch.no_grad():
                X_, Z, latent_dist = model.forward_(X)
        elif isinstance(model, jointvae.JointVAE):
            with torch.no_grad():
                X_, Z, latent_dist = model.forward_(X)
                latent_spec = model.config['latent_spec']
                cont_dim = latent_spec['cont']
                disc_dims = latent_spec['disc']
                cont_Z = Z[:,:cont_dim]
                disc_Zs = [Z[:,cont_dim + dim1:cont_dim + dim1 + dim2] for dim1, dim2 in zip([0] + disc_dims, disc_dims)]
                disc_Zs = [np.argmax(disc_Z, axis=1).unsqueeze(dim=1) for disc_Z in  disc_Zs]
                Z = torch.hstack((cont_Z, *disc_Zs))
        else:
            raise RuntimeError('Unsupported model for disentanglement metric computation')
        return Z.cpu().detach().numpy()


def main(argc, argv):
    # Set and parse arguments:
    parser = argparse.ArgumentParser(description='Compute disentanglement metrics for models located in model_base_path directory, listed in CSV file.')
    parser.add_argument('--session', type=str, help='Session number, used to identify model database with the process', default='01')
    parser.add_argument('--model-base-path', type=str, help='Full name of model', 
        default='/home/krzyja/Workspace/SpecVAE/.model/HMDB/jointvae_capacity')
    parser.add_argument('--csv-filepath', type=str, help='Path to csv file with models', 
        default='/home/krzyja/Workspace/SpecVAE/.model/HMDB/jointvae_capacity/experiment01.csv')
    parser.add_argument('--n-samples', type=int, help='Number of samples to be preloaded in total', default=15000)
    parser.add_argument('--batch-size', type=int, help='Number of points to be used to compute the training_sample', default=128)
    parser.add_argument('--n-train-samples', type=int, help='Number of points used for training', default=10000)
    parser.add_argument('--n-eval-samples', type=int, help='Number of points used for evaluation', default=5000)
    parser.add_argument('--n-variance-estimate-samples', type=int, help='Number of points used to estimate global variances', default=5000)
    parser.add_argument('--n-proc', type=int, help='Number of processes to compute DMs score on', default=1)
    parser.add_argument('--debug', type=utils.boolean_string, help='Run in debug mode on single thread', default=False)
    parser.add_argument('--n-chunks', type=int, help='Number of chunks to split database into', default=10)
    args = parser.parse_args()

    # Parameters:
    session                     = args.session
    model_base_path             = args.model_base_path
    filepath                    = args.csv_filepath
    n_samples                   = args.n_samples
    batch_size                  = args.batch_size
    n_train_samples             = args.n_train_samples
    n_eval_samples              = args.n_eval_samples
    n_variance_estimate_samples = args.n_variance_estimate_samples
    n_proc                      = args.n_proc
    debug                       = args.debug
    n_chunks                    = args.n_chunks

    # Configure log file:
    import os
    base_path = utils.get_project_path() / '.out' / 'dms'
    logfilepath = base_path / ('dm_%s.log' % datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    os.makedirs(str(base_path), exist_ok=True)
    logging.basicConfig(
        filename=str(logfilepath), 
        format='[%(levelname)s] %(asctime)s (%(threadName)s): %(message)s', 
        level=logging.DEBUG)

    # Verify paths:
    if not os.path.exists(model_base_path):
        logging.error("Directory '%s' doesn't exist" % model_base_path)
        return 1
    if not os.path.exists(filepath):
        logging.error("File '%s' doesn't exist" % filepath)
        return 1

    try:
        dm = DisentanglementMetric(model_base_path, n_samples, batch_size, 
            n_train_samples, n_eval_samples, n_variance_estimate_samples)

        # Read and process CSV file:
        logging.info("Start computing DMs for configuration: %s" % str(args))
        df1 = pd.read_csv(filepath, index_col=0, error_bad_lines=False)
        df_ = np.array_split(df1.copy(), n_chunks)

        # Read chunk by chunk:
        for i, df in enumerate(df_):
            logging.info("Start computing %d/%d chunk" % (i+1, n_chunks))

            if not debug:
                with mp.Pool(n_proc) as pool:
                    results = pool.map(dm.compute, df['full_model_name'])
            else:
                model_name_list = df['full_model_name'].tolist()
                results = []
                for full_model_name in model_name_list:
                    results.append(dm.compute(full_model_name))

            extracted = [
                {
                    'm.' + k + '.' + kk + '.' + str(kkk): vvv 
                        for k, v in result.items() 
                            for kk, vv in v.items() 
                                for kkk, vvv in vv.items()
                } for result in results
            ]

            # Save results in a copy of csv file:
            path, filename = os.path.split(filepath)
            newfilename = '%s_dms.csv' % os.path.splitext(filename)[0]
            new_filepath = os.path.join(path, newfilename)

            try:
                # Save results:
                df_ext = pd.DataFrame(extracted)
                df_ext.index = df.index
                df_new = pd.concat([df, df_ext], axis=1)
                if os.path.exists(new_filepath):
                    df_saved = pd.read_csv(new_filepath, index_col=0)
                    df_saved = pd.concat([df_saved, df_new], ignore_index=True)
                    df_saved.to_csv(new_filepath)
                else:
                    df_new.to_csv(new_filepath)

                logging.info("Saved chunk %d/%d" % (i+1, n_chunks))

            except Exception as e:
                logging.exception("Error while adding records to '%s'" % new_filepath)

    except Exception as e:
        logging.exception("Error while computing DMs for '%s' file" % filepath)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
