#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

DATASET_NAME="sft_25k"
# DATASET_NAME="try_debug"

# read from .bashrc or set explicitly
WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"

# echo "Download SFT data and videos. Update dataset_info.json"
# echo "Current SFT data name: ${DATASET_NAME}"
# python examples/train_full/prepare_data.py \
#     --sft_data_name "sft_25k" \
#     --frame_or_video "v"

wandb login --relogin $WANDB_API_KEY
echo "WandB login."

echo "Start training ..."
CUDA_VISIBLE_DEVICES=1,2,3,4 llamafactory-cli train examples/train_full/vs2_qwen2_5vl_sft_25k_2e-4_2fps_960_720_8192.yaml \
    hf_hub_token=${HF_TOKEN} \
    dataset=${DATASET_NAME} 
