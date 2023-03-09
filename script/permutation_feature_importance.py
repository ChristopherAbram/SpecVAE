import sys
import pandas as pd
import argparse
import logging
import multiprocessing as mp
from datetime import datetime

import specvae.utils as utils
from specvae.metrics import PFI


def main(argc, argv):
    # Set and parse arguments:
    parser = argparse.ArgumentParser(description='Compute permutation feature importance for models located in model_base_path directory, listed in CSV file.')
    parser.add_argument('--session', type=str, help='Session number, used to identify model database with the process', default='01')
    parser.add_argument('--model-base-path', type=str, help='Full name of model', 
        default='/home/krzyja/Workspace/SpecVAE/.model/HMDB/betavae_reg')
    parser.add_argument('--csv-filepath', type=str, help='Path to csv file with models', 
        default='/home/krzyja/Workspace/SpecVAE/.model/HMDB/betavae_reg/experiment01.csv')
    parser.add_argument('--n-repeats', type=int, help='Number of times to permute a feature', default=10)
    parser.add_argument('--n-samples', type=int, help='Number of samples used for PFI computation', default=3000)
    parser.add_argument('--n-proc', type=int, help='Number of processes to compute PFI score on', default=6)
    args = parser.parse_args()

    # Parameters:
    session             = args.session
    model_base_path     = args.model_base_path
    filepath            = args.csv_filepath
    n_repeats           = args.n_repeats
    n_samples           = args.n_samples
    n_proc              = args.n_proc

    # Configure log file:
    import os
    base_path = utils.get_project_path() / '.out' / 'feature_importance'
    logfilepath = base_path / ('pfi_%s.log' % datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
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
        # Create PFI metric:
        pfi = PFI(model_base_path, n_samples, n_repeats)

        # Read and process CSV file:
        logging.info("Start computing PFI for configuration: %s" % str(args))
        df = pd.read_csv(filepath, index_col=0, error_bad_lines=False)
        pool = mp.Pool(n_proc)
        df1 = df.copy()
        df1['feature_importance'] = pool.map(pfi.compute_pfi, df1['full_model_name'])

        # Save results in a copy of csv file:
        path, filename = os.path.split(filepath)
        newfilename = '%s_pfi.csv' % os.path.splitext(filename)[0]
        df1.to_csv(os.path.join(path, newfilename))

    except Exception as e:
        logging.exception("Error while computing PFI for '%s' file" % filepath)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
