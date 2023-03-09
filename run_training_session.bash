#!/bin/bash
mkdir -p .out
source .venv/bin/activate
python3 script/train_supervisor.py --config-file $1 > /dev/null 2>&1 &
disown
