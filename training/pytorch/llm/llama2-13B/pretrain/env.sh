#!/bin/bash

# Copyright © 2024 Shanghai Biren Technology Co., Ltd. All rights reserved.

if command -v brcc >/dev/null 2>&1; then
  echo 'found brcc, set BR envs'

  bash ${_THIS_DIR}/check_gpu_free.sh || exit -1

  ulimit -n 1048576 #linux 
  export OMP_NUM_THREADS=16
  # br_pytorch envs
  export BRTB_PLAN_ID_RENEW=1
  export BRTB_DISABLE_ZERO_REORDER=1
  export BRTB_DISABLE_ZERO_OUTPUT_NUMA=1
  export BRTB_DISABLE_ZERO_OUTPUT_UMA=1
  export BRTB_DISABLE_ZERO_WS=1
  export BRTB_ENABLE_FORCE_UMA=1
  export BRTB_ENABLE_SUPA_FILL=1
  export BRTB_ENABLE_SUBLAS_API=1
  export BRTB_ENABLE_MMA_BF16_ACC=1 # do mma acc by bf16, involved in sublas linear and mlp inner mma, faster but lower precision, disable for MLFW-3678
  export BRTB_ENABLE_REGISTER_BEFORE_D2H=1
  # export BRTB_DISABLE_DYNAMIC=1
  # export BRTB_LOG_LEVEL=debug
  # export BRTB_LOG_BACKEND=stdout
  # export BRTB_SUPA_KER_SELECTION_DEBUG=1
  # export BRTB_PERF_REPORT=1

  # memory
  export PYTORCH_SUPA_ALLOC_CONF=max_host_pinned_memory_size_mb:102400,max_split_size_mb:32

  total_mem=`cat /sys/class/biren/card_0/device/mem_info|grep "MemTotal" | head -1|awk '{print $2}'`
  if  [ $total_mem != "33292288" ];then
    chroot /mnt/host-path/ /bin/bash -c "sudo rmmod biren && sudo modprobe biren biren_health_check=0 biren_res_hbm_size=256"
  fi
  # umd envs
  # export BR_UMD_TRACE_LEVEL=1
  # export BR_UMD_TRACE_EXCEPTION=1
  export BR_UMD_DEBUG_P2P_ACCESS_CHECK=0

  # sccl envs
  export SUCCL_BUFFSIZE=16777216
  export SUCCL_PARALLEL_NUM=3
  # export SUCCL_DEBUG=TRACE
  # export SUCCL_DEBUG_SUBSYS=ALL
  # export SUCCL_DEBUG_FILE=succl_debug_${NODE_RANK}_%h_%d.log
  # export SUCCL_SYNC_MODE=1
  # export SUPA_VISIBLE_DEVICES=4
  # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/SCCL/build
   
  # log
  export BRTB_LOG_LEVEL=Notice
  export SULIB_LOG_LEVEL=off
  export BRTB_LOG_BACKEND=empty
  # export SUBLAS_LOG_LEVEL=all
  # Megatron-LM log level, default=info. opt: debug/info/warning/error
  export FW_MGLM_LOG_LEVEL='info'

  #kernel_cache
  export SUDNN_KERNEL_CACHE_FOLDER=/tmp/kernel_cache/
  mkdir -p $SUDNN_KERNEL_CACHE_FOLDER
  export SUDNN_KERNEL_CACHE_CAPACITY=50000
  export SUDNN_KERNEL_CACHE_DISK_LEVEL=3
  export SUDNN_KERNEL_CACHE_EXCLUDE_UID=1
  export SUDNN_KERNEL_CACHE_MAX_SIZE_MB=10240
  export SUDNN_KERNEL_CACHE_THRESHOLD=0

else
  echo 'No brcc found'
  exit
fi
export NCCL_DEBUG=warn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=14

START_DATE=$(date "+%Y_%m_%d_%H:%M:%S")
timestamp=$(date +%s)
echo "START TIME: $START_DATE"

mkdir -p $OUTPUT_DIR

EXPNAME=${EXPNAME:-experiment}
variant=${EXPNAME}_seqlen${SEQ_LEN}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_l${NLAYERS}_tp${TP_SIZE}_pp${PP_SIZE}_n${NNODES}

DATA_OUTPUT_PATH=$OUTPUT_DIR/model_dir/llama2_13b_zh
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints/$variant
REPO_PATH=$DATA_OUTPUT_PATH/experiment
TENSORBOARD_PATH=$REPO_PATH/tensorboard/$variant/${timestamp}
LOGS_PATH=$REPO_PATH/logs/$variant/${timestamp}/$NODE_RANK
mkdir -p $LOGS_PATH

KILL_SWITCH_PATH=$REPO_PATH/kill-switch

export | grep "ENABLE\|DISABLE\|BRTB\|SUDNN\|SULIB\|TORCH\|SUCCL\|SUPA\|NV\|CUDA"