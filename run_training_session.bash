#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
mkdir -p .out
conda activate specvae
python3 specvae/train_supervisor.py --config-file $1 > /dev/null 2>&1 &
disown
