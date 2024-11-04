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
from vllm import LLM, SamplingParams

# 模型配置文件路径
config_path = "configs/qwen2_72b.json"
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = [SamplingParams(n=1, temperature=0.9, top_p=0.95, max_tokens=512)] * len(prompts)
llm = LLM(config_path=config_path, trust_remote_code=True)
outputs = llm.generate(prompts, sampling_params)

# 打印输出结果
for output in outputs:
    prompt = output.prompt
    for sample in output.outputs:
        print(f"Prompt: {prompt!r}, Generated text: {sample.text!r}")