SUDNN_KERNEL_CACHE_CAPACITY=30000 \
SUDNN_KERNEL_CACHE_MAX_SIZE_MB=1024 \
SUDNN_KERNEL_CACHE_THRESHOLD=0 \
SUDNN_KERNEL_CACHE_EXCLUDE_UID=1 \
SUDNN_KERNEL_CACHE_DISK_LEVEL=3 \
SUPA_DEVICE_ORDER=PCI_BUS_ID \
BRTB_ENABLE_EAGER_CAT=1 \
BRTB_ENABLE_SUPA_FILL=1 \
BRTB_DISABLE_L2_FLUSH=1 \
BRTB_DISABLE_ZERO_REORDER=1 \
python3 test.py            \
	--data coco.yaml   \
	--batch-size 32    \
	--img-size 640     \
	--conf 0.001       \
       	--iou 0.65         \
       	--device supa      \
	--weights 'weights/yolov5s.pt'
