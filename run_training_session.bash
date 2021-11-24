#!/bin/bash
mkdir -p .out/
python3 specvae/train_vae.py > .out/out8.txt 2>&1 &
disown