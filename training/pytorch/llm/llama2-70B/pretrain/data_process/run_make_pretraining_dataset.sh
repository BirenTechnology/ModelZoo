#! /bin/bash
START_TIME=$SECONDS

INPUT_JSON=$1
TOKENIZER_DIR=$2
OUTPUT_DIR=$3

mkdir -p $OUTPUT_DIR
OUTPUT_PREFIX=$OUTPUT_DIR/redpajama-llama2
python3 -u tools/preprocess_data.py \
    --input $INPUT_JSON \
    --output-prefix $OUTPUT_PREFIX \
    --tokenizer-type Llama2Tokenizer \
    --vocab-file $TOKENIZER_DIR \
    --tokenizer-model $TOKENIZER_DIR/tokenizer.model \
   --append-eod \
    --workers 180 \
    --partitions 10

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"