#!/bin/bash

# Copyright © 2024 Shanghai Biren Technology Co., Ltd. All rights reserved.

if command -v brcc >/dev/null 2>&1; then
  echo 'found brcc, set BR envs'

  bash ${_THIS_DIR}/check_gpu_free.sh || exit -1

  # br_pytorch envs
  export BRTB_PLAN_ID_RENEW=1
  export BRTB_DISABLE_ZERO_REORDER=1
  export BRTB_DISABLE_ZERO_OUTPUT_NUMA=1
  export BRTB_DISABLE_ZERO_OUTPUT_UMA=1
  export BRTB_DISABLE_ZERO_WS=1
  export BRTB_ENABLE_FORCE_UMA=1
  export BRTB_ENABLE_SUPA_FILL=1
  export BRTB_ENABLE_SUBLAS_API=1
  export BRTB_ENABLE_SUPA_ATTENTION=1
  export BRTB_ENABLE_WEIGHT_BYPASS=0
  export BRTB_ENABLE_MMA_BF16_ACC=0
  export BRTB_ENABLE_REGISTER_BEFORE_D2H=1
  export BRTB_LEGACY_PROFILER_STACK=1
  export BRTB_ENABLE_ROWMAJOR_TENSOR=1
  export PYTORCH_SUPA_ALLOC_CONF=max_split_size_mb:128
  export SULIB_LOG_LEVEL=off

  # umd envs
  # export BR_UMD_TRACE_LEVEL=1
  # export BR_UMD_TRACE_EXCEPTION=1
  export BR_UMD_DEBUG_P2P_ACCESS_CHECK=0 
  export BR_UMD_DEBUG_DEVICE_RING_BUFFER=1

  # sccl envs
  export SUCCL_BUFFSIZE=134217728
  export SUCCL_PARALLEL_NUM=5
  # export TRACE=1
  # export SUCCL_DEBUG=TRACE
  # export SUCCL_DEBUG_SUBSYS=ALL
  # export SUCCL_DEBUG_FILE=succl_debug_%h_%d.log

  # interleaved 1f1b
  export SUCCL_NO_MEMCPY=1
  export BRTB_DIST_ENABLE_MULTI_STREAM=1

  #kernel_cache
  export SUDNN_KERNEL_CACHE_FOLDER=/tmp/kernel_cache/
  mkdir -p $SUDNN_KERNEL_CACHE_FOLDER
  export SUDNN_KERNEL_CACHE_CAPACITY=50000
  export SUDNN_KERNEL_CACHE_DISK_LEVEL=3
  export SUDNN_KERNEL_CACHE_EXCLUDE_UID=1
  export SUDNN_KERNEL_CACHE_MAX_SIZE_MB=10240
  export SUDNN_KERNEL_CACHE_THRESHOLD=0
   
else
  echo 'No brcc found, set NV envs'
  export NV_LIBNCCL_DEV_PACKAGE=
  export NV_LIBNCCL_DEV_PACKAGE_VERSION=
  export NV_LIBNCCL_DEV_PACKAGE_NAME=
  export NV_LIBNCCL_PACKAGE=
  export NV_LIBNCCL_PACKAGE_NAME=
  export NV_LIBNCCL_PACKAGE_VERSION=
  # export NCCL_SOCKET_IFNAME=eth0

  # 单节点 bug
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

  # do not remove or the training will hang and nodes will be lost w/o this workaround
  export CUDA_LAUNCH_BLOCKING=1

  # force crashing on nccl issues like hanging broadcast
  export NCCL_ASYNC_ERROR_HANDLING=1
fi

export NCCL_DEBUG=warn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=14

START_DATE=$(date "+%Y_%m_%d_%H:%M:%S")
timestamp=$(date +%s)
echo "START TIME: $START_DATE"

EXPNAME=${EXPNAME:-default_exp_name}
variant=mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_l${NLAYERS}_tp${TP_SIZE}_pp${PP_SIZE}_n${NNODES}_${EXPNAME}

ML_OUTPUT_DIR=${ML_OUTPUT_DIR:-./output}
mkdir -p $ML_OUTPUT_DIR

DATA_OUTPUT_PATH=$ML_OUTPUT_DIR/model_dir/llama2_70b_zh
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/experiment
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant/${timestamp}
LOGS_PATH=$REPO_PATH/logs/$variant/$NODE_RANK
mkdir -p $LOGS_PATH
mkdir -p $TENSORBOARD_PATH
KILL_SWITCH_PATH=$REPO_PATH/kill-switch

DATA_PARENT_PATH=${DATA_PARENT_PATH:-/mnt/host-path/share/datasets/official/RedPajama-Data-1T}
DATA_PATH="$DATA_PARENT_PATH/redpajama-llama2_text_document"
VOCAB_FILE=$_THIS_DIR/../../tokenizer
TOKENIZER_MODEL=$_THIS_DIR/../../tokenizer/tokenizer.model

export | grep "ENABLE\|DISABLE\|BRTB\|SUDNN\|SULIB\|TORCH\|SUCCL\|SUPA\|NV\|CUDA"
