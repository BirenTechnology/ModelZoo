#!/bin/bash

export SUDNN_KERNEL_CACHE_CAPACITY=30000
export SUDNN_KERNEL_CACHE_MAX_SIZE_MB=10240
export SUDNN_KERNEL_CACHE_EXCLUDE_UID=1
export SUDNN_KERNEL_CACHE_DISK_LEVEL=3
export SUDNN_KERNEL_CACHE_DEVICE_MEMORY_MAX_SIZE_MB=10240
export BRTB_KERNEL_CACHE_CAPACITY=30000
export BRTB_USE_SUDNN_BN_FWD=1
export BRTB_ENABLE_SUDNN_BN_BWD=1
export BRTB_ENABLE_EAGER_CAT=1
export SUDNN_KERNEL_CACHE_FOLDER=/root/yolo_kernel_cache
export BRTB_DISABLE_L2_FLUSH=1
export BRTB_DISABLE_ZERO_REORDER=1
export BRTB_DISABLE_ZERO_OUTPUT_NUMA=1
export BRTB_DISABLE_ZERO_WS=1
export BRTB_DISABLE_ZERO_OUTPUT_UMA=1
export ENABLE_CLEAN_TENSOR=1
export BRTB_ENABLE_NUMA_SPLIT=1

yolo detect train data=coco.yaml model=yolov8l.pt epochs=100 imgsz=640 device="0, 1, 2, 3, 4, 5, 6, 7"
