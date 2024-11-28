# Copyright 2024 Shanghai Biren Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import json
import argparse
from tqdm import tqdm

seed=42
os.environ['PYTHONHASHSEED'] = str(seed)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input_dir', type=str, required=True,
                       help='Path to input folder')
    group.add_argument('--output_json', type=str, default='./out_put.json',
                       help='Path to JSON output file')
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    directory = os.path.dirname(args.output_json)

    if not os.path.exists(directory):
        os.makedirs(directory)

    combined_df = pd.DataFrame()    
    file_res= []
    for file in os.listdir(args.input_dir):
        file_res.append(file)
    
    final_file_res = sorted(file_res,key=lambda x:x.encode("UTF-8"))

    for file_name  in tqdm(final_file_res):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(args.input_dir, file_name)
            df = pd.read_parquet(file_path)
            combined_df = pd.concat([combined_df, df])
    print("Saving ... Please do not exit ...")
    combined_df.to_json(args.output_json, orient='records', lines=True, force_ascii=False)
    print("Processing completed")

if __name__ == '__main__':
    
    main()