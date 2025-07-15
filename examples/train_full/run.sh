#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

HUB_USER_NAME="videoscore2_exp"
CURR_MODEL_NAME="vs2_qwen2_5vl_sft_17k"
DATASET_NAME="sft_17k"
HUB_MODEL_ID="${HUB_USER_NAME}/${CURR_MODEL_NAME}"

# read from .bashrc or set explicitly
WANDB_TOKEN="${WANDB_TOKEN:?Set WANDB_TOKEN}"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"

echo "Download SFT data and videos. Update dataset_info.json"
echo "Current SFT data name: ${DATASET_NAME}"
python examples/train_full/prepare_data.py \
    --sft_data_name ${DATASET_NAME}

wandb login --relogin $WANDB_TOKEN
echo "WandB login."

echo "Start training ..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwen2_5vl_7b_full_sft.yaml \
    dataset=${DATASET_NAME} \
    output_dir="saves/${CURR_MODEL_NAME}" \
    export_hub_model_id=${HUB_MODEL_ID} \
    hf_hub_token=${HF_TOKEN} \

