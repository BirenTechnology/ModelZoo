#!/bin/bash
# Copyright © 2023 Shanghai Biren Technology Co., Ltd. All rights reserved.

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps $ADAM_EPS \
    --lr $LR\
    --min-lr $MIN_LR \
    --train-iters $MAX_TRAINING_ITERS \
    --lr-decay-style cosine \
    --lr-decay-iters 320000 \
    --lr-warmup-iters $LR_WARMIP_ITERS \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "
# for 20h 1190, for 100h 5990
#    --exit-duration-in-mins 1190
# EXIT_OPTS=" \
#     --exit-duration-in-mins 99999999 \
#     "

MODEL_ARGS=" \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --num-query-groups $NQG \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $MAX_POS \
    --tokenizer-type "Llama2Tokenizer" \
    --tokenizer-model $TOKENIZER_MODEL \
    --loss-scale $LOSS_SCALE \
    --init-method-std 0.0048 \
    --seed $SEED \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --bf16 \
    --distributed-backend nccl \
    --use-cpu-initialization \
    --disable-bias-linear \
    --transformer-impl local \
    --no-create-attention-mask-in-dataloader \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
"

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 0 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --profiler-schedule 1 1 1 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    "

DATA_ARGS=" \
    --data-path $DATA_PATH \
"

DISTRIBUTED_ARGS=" \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"
