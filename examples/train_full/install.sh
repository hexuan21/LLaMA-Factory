#!/bin/bash

set -x

pip install -e ".[torch,metrics]" --no-build-isolation
pip install wandb
pip install huggingface_hub