#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

DATASET_NAME="sft_17k"
# DATASET_NAME="try_debug"

# read from .bashrc or set explicitly
WANDB_TOKEN="${WANDB_TOKEN:?Set WANDB_TOKEN}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"

# echo "Download SFT data and videos. Update dataset_info.json"
# echo "Current SFT data name: ${DATASET_NAME}"
# python examples/train_full/prepare_data.py \
#     --sft_data_name ${DATASET_NAME} \
#     --frame_or_video "v"

wandb login --relogin $WANDB_TOKEN
echo "WandB login."

echo "Start training ..."
llamafactory-cli train examples/train_full/vs2_qwen2_5vl_sft_17k_1e-5_4fps_8192.yaml \
    hf_hub_token=${HF_TOKEN} \
    dataset=${DATASET_NAME} 
