#!/bin/sh

export DATA_NAME="flower_dataset"
export EXP_NAME="best_hyperparam"

export NUM_WORKERS=0
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS_PER_NODE=1

export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1235


sh scripts/train.sh
# sh scripts/eval_multiGPU.sh
# sh scripts/search.sh
# sh scripts/search_multiGPU.sh