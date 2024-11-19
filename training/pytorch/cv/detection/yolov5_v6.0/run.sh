# kernel cache
export SUDNN_KERNEL_CACHE_CAPACITY=30000
export SUDNN_KERNEL_CACHE_MAX_SIZE_MB=10240
export SUDNN_KERNEL_CACHE_EXCLUDE_UID=1
export SUDNN_KERNEL_CACHE_DISK_LEVEL=3
export BRTB_KERNEL_CACHE_CAPACITY=30000
export BRTB_USE_SUDNN_BN_FWD=1
export BRTB_ENABLE_SUDNN_BN_BWD=1
export BRTB_ENABLE_EAGER_CAT=1
export SUDNN_KERNEL_CACHE_FOLDER=/root/v5s-kernel-cache-64
export BRTB_DISABLE_ZERO_REORDER=1
export BRTB_DISABLE_ZERO_OUTPUT_NUMA=1
export BRTB_DISABLE_ZERO_WS=1
export BRTB_DISABLE_ZERO_OUTPUT_UMA=1
export ENABLE_CLEAN_TENSOR=1
export PYTHONPATH=/root/br_pytorch/build/Release/packages:$PYTHONPATH
python3 train.py --data coco.yaml --cfg yolov5s.yaml --weights "" --batch-size 64 --device supa  --workers 8
