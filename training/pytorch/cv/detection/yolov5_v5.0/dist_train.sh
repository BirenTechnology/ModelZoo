env -i 
export BRTB_ENABLE_EAGER_CAT=1
export BRTB_USE_SUDNN_BN_FWD=1
export BRTB_ENABLE_SUDNN_BN_BWD=1
export BRTB_DISABLE_ZERO_REORDER=1
export SUDNN_KERNEL_CACHE_FOLDER=./kernel-cache
export SUDNN_KERNEL_CACHE_CAPACITY=2000
export SUDNN_KERNEL_CACHE_EXCLUDE_UID=1
export SUDNN_KERNEL_CACHE_MAX_SIZE_MB=2048
export SUDNN_KERNEL_CACHE_THRESHOLD=0
export BRTB_DISABLE_ZERO_OUTPUT_NUMA=1
export BRTB_DISABLE_ZERO_WS=1
export BRTB_DISABLE_ZERO_OUTPUT_UMA=1
export ENABLE_CLEAN_TENSOR=1
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12346 \
    train.py \
    --cfg yolov5m.yaml \
    --weight ' ' \
    --device supa \
    --data data/coco.yaml \
    --workers 6 \
    --batch-size 256 \
    --use_async_dataloader
