import sys, os
import argparse
from specvae import utils
import itertools as it
import json
import threading
import subprocess
import logging


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in it.product(*dicts.values()))

def run_session(parameters, config):
    path = utils.get_project_path()
    name = config['name']
    script = config['script']
    session_name = parameters['--session']

    # Prepare parameters:
    non_array_params = {name: value for name, value in parameters.items() if not isinstance(value, list) and not isinstance(value, dict)}
    array_params = {name: value for name, value in parameters.items() if isinstance(value, list)}
    dict_params = {name: value for name, value in parameters.items() if isinstance(value, dict)}
    args_product = list(dict_product(array_params))
    args_len = len(args_product)
    arg_list_one = []
    for k, v in non_array_params.items():
        arg_list_one += [k, str(v)]

    # Create paired dict configs:
    # TODO: if there are 2 or more dicts, it should generate all combinations dict1 x dict2 x ...
    arg_list_3 = []
    for param, d in dict_params.items():
        key_param = param
        value_param = d[param]
        del d[param]
        for k, v in d.items():
            arg_list_3.append([key_param, k, value_param, str(v)])

    arg3_len = len(arg_list_3)
    logging.info("[%s] Run session, items: %d" % (session_name, args_len * (arg3_len if arg3_len > 0 else 1)))

    # # Load experiment csv file:
    # df = None
    # if resume_setting is not None:
    #     try:
    #         import pandas as pd
    #         import os
    #         filepath = utils.get_project_path() / '.model' / dataset / name / ('experiment%s.csv' % session_name)
    #         if os.path.exists(filepath):
    #             logging.info("[%s] Resume session from file: %s" % (session_name, str(filepath)))
    #             df = pd.read_csv(filepath, index_col=False, error_bad_lines=False)
    #         else:
    #             logging.info("[%s] Session file doesn't exist, run without resume" % session_name)

    #     except Exception as e:
    #         print("Error", e)

    if arg3_len == 0:
        arg_list_3.append([])

    ex_count, err_count = 0, 0
    with open(str(path / '.out' / name / ('error%s.txt' % session_name)), 'w') as err:
        for dict_arg in arg_list_3:
            for args in args_product:
                # Skip if configuration have already been trained:
                # args_ = {**non_array_params, **args}
                # query_array = []
                # for arg_name, param_name in resume_setting.items():
                #     value = args_[arg_name]
                #     if param_name == 'layer_config':
                #         if '$indim' in value:
                #             # 
                #             value = value.replace('$indim', 2*args_['--n-peaks'])
                #     if isinstance(value, str):
                #         query_array.append('`%s` == "%s"' % (param_name, value))
                #     else:
                #         query_array.append('`%s` == %s' % (param_name, str(value)))
                # query = ' and '.join(query_array)
                # if df is not None:
                #     pass
                
                arg_list_two = []
                for k, v in args.items():
                    arg_list_two += [k, str(v)]

                arg_list = arg_list_one + arg_list_two + dict_arg
                logging.info("[%s] Run configuration: %s" % (session_name, str(arg_list)))
                python_script_path = path / script
                try:
                    with open(str(path / '.out' / name / ('out%s.txt' % session_name)), 'w') as out:
                        cp = subprocess.run(
                            ['python', python_script_path] + arg_list,
                            text=True, shell=False, stdout=out, stderr=err)

                        if cp.returncode != 0:
                            err_count += 1
                            logging.info("[%s] Error for configuration: %s" % (session_name, str(arg_list)))

                except Exception as e:
                    ex_count += 1
                    logging.error(e, exc_info=True)

    if err_count == 0 and ex_count == 0:
        logging.info("[%s] Session finished %d tasks without errors" % (session_name, args_len))
    else:
        logging.info("[%s] Session finished %d tasks with %d errors and %d exceptions" % (session_name, args_len, err_count, ex_count))
        
    return 0
    

def main(argc, argv):
    # Set parameters and parse:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--config-file', type=str, 
        help='Path to training session json file', 
        default=str(utils.get_project_path() / '.train' / 'jointvae_mona.json'))
    args = parser.parse_args()

    # Parameters:
    filepath = args.config_file
    config = {}
    with open(filepath, 'r') as file:
        config = json.load(file)

    # Open session log file:
    import os
    logdir = utils.get_project_path() / '.out' / config['name']
    logfilepath = logdir / 'out.supervisor.log'
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        filename=logfilepath, 
        format='[%(levelname)s] %(asctime)s (%(threadName)s): %(message)s', 
        level=logging.INFO)

    # Per each session create a thread:
    ths = []
    for session in config['sessions']:
        parameters = {
            **session,
            **config['parameters']
        }
        ths.append(threading.Thread(
            target=run_session, args=(parameters, config,)))

    # Start all sessions:
    for th in ths:
        th.start()

    # Wait to complete all sessions:
    for th in ths:
        th.join()

    logging.info("DONE!")
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
