#!/bin/bash

# Copyright © 2024 Shanghai Biren Technology Co., Ltd. All rights reserved.
env -i
_THIS_DIR=$(dirname "$0")

ulimit -n 1048576
mount -o remount,size=32G /dev/shm

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=$1 ## 多机训练需要将ip改成 NODE_RANK=0的机器ip
MASTER_PORT=60001 
NNODES=2 ## 训练机器数
NODE_RANK=$2 ## 机器编号 多机器训练需要修改成对应编号
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

EXPNAME=${EXPNAME:-multi_test}

## 日志文件输出路径
OUTPUT_DIR=$_THIS_DIR/output/

## 模型数据集路径
DATA_PARENT_PATH=/path/wikipedia
DATA_PATH="$DATA_PARENT_PATH/mmap_llama2_datasets_text_document"
if [ ! -d "$DATA_PARENT_PATH" ]; then
  echo "$DATA_PARENT_PATH not found"
  exit
fi

## tokenizer.model文件路径
TOKENIZER_MODEL=/path/tokenizer.model
if [ ! -e "$TOKENIZER_MODEL" ]; then
  echo "$TOKENIZER_MODEL not found"
  exit
fi

SAVE_INTERVAL=1000 ##每多少iter存一个ckpt
MAX_TRAINING_ITERS=${MAX_TRAINING_ITERS:-3000} ## 最大训练iters数目，执行完整流程后退出

INTERVAL_NUM=$(expr $MAX_TRAINING_ITERS + 1)
EVAL_INTERVAL=${EVAL_INTERVAL:-$INTERVAL_NUM} # 每xx个iter做一次eval
EIXT_INTERVAL=${EIXT_INTERVAL:-$INTERVAL_NUM} ## 第iter提前退出

# 单机八卡训练最佳性能策略
TP_SIZE=4
PP_SIZE=4
GLOBAL_BATCH=${GB:-1024} ##
MICRO_BATCH=4

# 训练超参数
ADAM_EPS=1e-8 ##
LR=1e-4  ##
MIN_LR=1e-5 ## 
LR_WARMIP_ITERS=500 ## 200
SEED=42 ## 1234
LOSS_SCALE=12 ##

# 模型结构
NLAYERS=40 ## 模型层数，需要除PP_SIZE
HIDDEN=5120
FFN_HIDDEN_SIZE=13824
SEQ_LEN=${SEQ_LEN:-2048}
MAX_POS=$SEQ_LEN
NHEADS=40
NQG=1

source $_THIS_DIR/env.sh
source $_THIS_DIR/args.sh

# 不同seq_len下显存占用存在差异，需要使用不同的训练策略达到最佳性能
if [ "$SEQ_LEN" -eq 2048 ]; then
PERF_ARGS=" \
    --supa-attention-v1 \
    --cpu-optimizer \
    --recompute-granularity=full \
    --recompute-method=block \
    --recompute-num-layers 1 \
    --recompute-num-layers-of-stages 3 1 0 0 \
    "
elif [ "$SEQ_LEN" -eq 4096 ]; then
PERF_ARGS=" \
    --sudnn-attention \
    --cpu-optimizer \
    --recompute-num-layers 1 \
    --recompute-granularity=full \
    --recompute-method=block \
    --recompute-num-layers-of-stages 4 2 1 0\
    "
else
  echo "unsupport SEQ_LEN=$SEQ_LEN, please use 2048 or 4096"
  exit
fi

LOCAL_ARGS=" \
    --position-embedding-type rope \
    --no-position-embedding \
    --untie-embeddings-and-output-weights \
    --use-rms-norm \
    --use-llama-mlp \
    --supa-fuse-embedding \
    --supa-fuse-rmsnorm \
    --supa-fuse-split-qkv \
    --supa-fuse-rope \
    --supa-fuse-attention-transform \
    --supa-fuse-crossentropy \
    --fused-mlp \
    --no-dropout \
    --sequence-parallel \
    --use-tensor-pool \
    --inplace-comm \
    --llama2mlp-with-tensor-pool \
    --num-workers 0 \
    --bf16-optimizer-use-flat-buffers \
    --bind-numa-node \
    --load ${CHECKPOINT_PATH} \
    --save ${CHECKPOINT_PATH} \
    --use-distributed-optimizer \
    --exit-interval $EIXT_INTERVAL \
    "

CMD="torchrun $DISTRIBUTED_ARGS ${_THIS_DIR}/pretrain_llama.py \
    $MODEL_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $PERF_ARGS \
    $LOCAL_ARGS \
    ${@:3} \
    "
bash -c "export&&${CMD}" 2>&1 | tee -a $LOGS_PATH/${timestamp}.log
echo log path: $LOGS_PATH/${timestamp}.log
echo tensorboard path: $TENSORBOARD_PATH
