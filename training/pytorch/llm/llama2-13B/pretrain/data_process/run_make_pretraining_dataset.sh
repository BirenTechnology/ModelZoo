#! /bin/bash
START_TIME=$SECONDS

input_json=$1          # 输入json
output_data_dir=$2     # 输出文件夹
tokenizer_model=$3     # 使用的tokenizer模型

INPUT="${input_json}"
TOKENIZER_MODEL="${tokenizer_model}"

python3 preprocess_data.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_llama2_datasets \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --json-keys 'text' \
  --workers 100 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"