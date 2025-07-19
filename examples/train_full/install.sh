#!/bin/bash

set -x

# conda create -n lmfac python=3.11 -y
# conda activate lmfac

pip install -e ".[torch,metrics]" --no-build-isolation
pip install wandb
pip install deepspeed==0.16.9
pip install --no-deps transformers==4.50.0