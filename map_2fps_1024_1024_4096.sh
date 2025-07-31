#!/bin/bash

source /map-vepfs/miniconda3/bin/activate
conda activate lmfac

export RANK=$MLP_ROLE_INDEX
export WORLD_SIZE=$(($MLP_WORKER_NUM * $MLP_WORKER_GPU))
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export LOCAL_RANK=$(($RANK % $MLP_WORKER_GPU))

unset https_proxy; unset http_proxy
export https_proxy="http://100.64.117.161:3128"
export http_proxy="http://100.64.117.161:3128"

cd /map-vepfs/xuan/LLaMA-Factory

export HF_HOME='/map-vepfs/huggingface'

wandb login --relogin $WANDB_TOKEN
echo "Start training..."
llamafactory-cli train examples/train_full/vs2_qwen2_5vl_sft_17k_2e-4_2fps_1024_1024_4096.yaml \
    hf_hub_token=$HF_TOKEN \
    dataset=sft_17k