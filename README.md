# SpecVAE

## Install Requirements
Follow below instructions to install the venv:
```bash 
# Create and activate venv
python -m venv .venv
source .venv/bin/activate

# Install python requirements
pip install -r requirements.txt
```

## Run Experiment
To run experiment, select the json file from the `.train` subdirectory (See [Experiment List](#experiment-list)) and execute:

```bash
# E.g. Train classification models on MoNA dataset:
source .venv/bin/activate
./run_training_session.bash .train/clf_mona.json
```

## Experiment List

| Name (JSON File) | Description |
| ---------------- | ----------- |
| `clf_mona` | Train deep classification models on the MoNA dataset using spectrum and another factors from the dataset. |
| `clf_hmdb` | Train deep classification models on the HMDB dataset using spectrum and another factors from the dataset. |
| `clf_mona_spectrum` | Train deep classification models on the MoNA dataset using only spectrum. |
| `clf_hmdb_spectrum` | Train deep classification models on the HMDB dataset using only spectrum. |
| `reg_mona` | Train deep regression models on the MoNA dataset using spectrum and another factors from the dataset. |
| `reg_hmdb` | Train deep regression models on the HMDB dataset using spectrum and another factors from the dataset. |
| `reg_mona_spectrum` | Train deep regression models on the MoNA dataset using only spectrum. |
| `reg_hmdb_spectrum` | Train deep regression models on the HMDB dataset using only spectrum. |
| `betavae_clf_mona` | Train deep classification models on the MoNA dataset using latent representations of spectrum and another factors from the dataset. |
| `betavae_clf_mona_latent` | Train deep classification models on the MoNA dataset using only latent representations of spectrum. |
| `betavae_clf_hmdb` | Train deep classification models on the HMDB dataset using latent representations of spectrum and another factors from the dataset. |
| `betavae_clf_hmdb_latent` | Train deep classification models on the HMDB dataset using only latent representations of spectrum. |
| `betavae_reg_mona` | Train deep regression models on the MoNA dataset using latent representations of spectrum and another factors from the dataset. |
| `betavae_reg_mona_latent` | Train deep regression models on the MoNA dataset using only latent representations of spectrum. |
| `betavae_reg_hmdb` | Train deep regression models on the HMDB dataset using latent representations of spectrum and another factors from the dataset. |
| `betavae_reg_hmdb_latent` | Train deep regression models on the HMDB dataset using only latent representations of spectrum. |
| `beta_vae_mona` | Train BetaVAE models on the MoNA or HMDB dataset using only spectrum. |
| `beta_vae_hmdb` | Train BetaVAE models on the MoNA or HMDB dataset using only spectrum. |
| `betavae_pfi` | Evaluate `permutation feature importance` metric on the regression and classification deep models trained jointly with the BetaVAE model, e.g. `betavae_clf_mona`, etc.  |
| `betavae_dms` | Evaluate `disentanglement` metrics on the BetaVAE models. |
| `jointvae_mona` | Train JointVAE models on the MoNA dataset. |
| `jointvae_hmdb` | Train JointVAE models on the HMDB dataset. |

### Structure of the JSON File
The JSON file defines the experiment, i.e. name, executable, parameter values, devices on which instances of the experiment run, etc. The experiment is then run by the `train_supervisor.py` which takes the path to the JSON experiment file.
The reasoning behind the `train_supervisor` is to have a lightweight task manager that can run multiple experiments, without introducing any additional dependencies.

The file consists of fields:
- `name` - the name of the experiment, it is used to create log files, and output directory in `.model/{dataset}/{name}`,
- `environment` - the name of the conda/venv environment (not used),
- `script` - the path to script that executes a single instance of the experiment,
- `parameters` - the set of parameters listed as key-value, if the value of a parameter is a list, then the supervisor will run the `script` for each item of the list. Note that multiple parameters can be lists, the supervisor will run the `script` for each combination of items from each list, i.e., the cartesian product of lists.
- `sessions` - the list of units of execution, i.e., specify how many conccurent jobs run simultaneously. Each session defines `--session`, the unique name of the session, and extend the list of `parameters` that are specific for given session, e.g., the first session schedule scripts on the GPU and the second on the CPU, schedule sessions on different GPU devices and each session has disjoint set of parameters (4 sessions partition the set of allowed combinations of parameters).

### Notes
Current design of the experiment file and the `train_supervisor` is an environment specific, i.e., can be reproduced only in specific environment, e.g., in case of the experiments in the above list, some of them can run only on machine with 4 GPUs.
However, it is still possible to reproduce experiments after redefining sessions.
