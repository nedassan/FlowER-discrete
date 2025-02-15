#!/bin/sh

torchrun \
  --node_rank="$NODE_RANK" \
  --nnodes="$NUM_NODES"\
  --nproc_per_node="$NUM_GPUS_PER_NODE" \
  --rdzv-id=456 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  beam_predict_multiGPU.py
