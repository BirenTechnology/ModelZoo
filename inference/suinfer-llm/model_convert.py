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
import torch
import argparse
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

def model_convert(args):
    if "chatglm" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, ignore_mismatched_sizes=True, trust_remote_code=True).float()
    elif "qwen2" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, attn_implementation="eager", trust_remote_code=True, torchscript=True, ignore_mismatched_sizes=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, ignore_mismatched_sizes=True)
    
    logger.info(f"The Replaced model: {model}")
    traced_model = torch.jit.trace(model, torch.randint(1, 10, (1, 10)), strict=False, check_trace=False)

    if not os.path.exists(args.output_dir):
        os.mkkdir(args.output_dir)
    torch.jit.save(traced_model, os.path.join(args.output_dir, args.model_name + ".pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Convert Tool.")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    model_convert(args)