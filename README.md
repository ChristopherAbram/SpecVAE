# SpecVAE

## Install requirements
Follow below instructions to install the venv:
```bash 
# Create and activate venv
python -m venv .venv
source .venv/bin/activate

# Install python requirements
pip install -r requirements.txt
```

## Run experiment
To run experiment, select the json file from the `.train` subdirectory and execute:

```bash
# E.g. Train classification models on MoNA dataset:
source .venv/bin/activate
./run_training_session.bash .train/clf_mona.json
```
