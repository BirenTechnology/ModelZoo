OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 3e-5 \
    --min-lr 3e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 282_478_095 \
    --lr-warmup-samples 313_864 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --train-samples 313_864_550 \
    "

EXIT_OPTS=" \
    --exit-interval ${EIXT_INTERVAL:-5} \
    "

MODEL_ARGS=" 
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NHEADS \
    --num-query-groups $NQG \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $MAX_POS \
    --train-samples 9_437_184 \
    --tokenizer-type "Llama2Tokenizer" \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-file $VOCAB_FILE \
    --loss-scale $LOSS_SCALE \
    --init-method-std 0.0048 \
    --seed 42 \
    --vocab-size 32128 \
    --ffn-hidden-size $FFN_SIZE \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --bf16 \
    --init-method-std 0.0048 \
    --distributed-backend nccl \
    --use-cpu-initialization \
    --disable-bias-linear \
    --weight-by-pass \
    --pad-vocab-size-to 32128 \
    --transformer-impl local \
    --apply-query-key-layer-scaling \
    --no-create-attention-mask-in-dataloader \
    --group-query-attention \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
"

OUTPUT_ARGS=" \
    --log-interval 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --eval-iters 10000 \
    --eval-interval 10000 \
    --save-interval 10000 \
    --tensorboard-dir $TENSORBOARD_PATH \
    "

DATA_ARGS="
    --data-path $DATA_PATH \
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"