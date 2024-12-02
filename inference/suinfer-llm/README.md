# 使用说明

## 环境依赖

* 操作系统：Ubuntu 20.04
* Python版本：Python3.8 Python3.10
* BIREN AI SDK
* suinfer-llm
* Python三方依赖
  * psutil
  * sentencepiece
  * numpy
  * transformers >= 4.40.0
  * fastapi == 0.111.0
  * uvicorn[standard]
  * pydantic == 2.7.1
  * prometheus_client>=0.18.0
  * torch==2.3.1
  * py-cpuinfo
  * outlines == 0.0.34
  
## 模型转换

### 转换环境

* 操作系统：Ubuntu 20.04
* Python版本：Python3.8
* 可根据Dockerfile 构建转换环境的docker,其中默认的transformers版本是4.40.1

> 注意：不同的模型需要的transformers的版本不同，推荐使用huggingface官方仓库中标明的版本。建议单独配置一个转换环境！

### 转换程序

从huggingface官网下载模型文件之后，需要把其中config.json文件中的use_cache字段设置为false，否则可能会导致转换失败。
模型转换成功之后，转换程序可参考model_convert.py，以Qwen/Qwen2-72B为例：

```bash
python3 model_convert.py --model_name models--Qwen--Qwen2-72B --model_dir /models--Qwen--Qwen2-72B/snapshots/87993795c78576318087f70b43fbf530eb7789e7 --output_dir ./
```

## 离线推理

### 配置文件构建

所有模型配置文件字段相同，并且均为json格式，这里以qwen2-72b为例进行介绍，其对应的配置文件为：
**qwen2_72b.json**

```json
{
  "model": "Qwen/Qwen2-72B",
  "cached_dir": "/models/qwen2-72b/models--Qwen--Qwen2-72B/snapshots/87993795c78576318087f70b43fbf530eb7789e7/",
  "input_name": "input_ids.1",
  "output_name": "9",
  "vocab_size": 152064,
  "model_file": "/models/qwen2-72b.pt",
  "serialize_file": "/root/qwen2_72b",
  "model_precision": "a_bf16_w_int8_kv_int8",
  "build_seq": 2048,
  "build_batch": 8,
  "distribute_param": [4, 1, 1],
  "devices": [0, 1, 2, 3],
  "all_models": {
    "http://host_ip:port": "Qwen/Qwen2-72B"
  }
}

```

在上述配置文件中：

* model：huggingface官网提供的完整模型名称。
* cached_dir：本地模型缓存路径，如果不设置这个字段，则启动服务时会根据model字段的内容从huggingface上下载模型文件。
* input_name：pt格式模型文件输入节点的名称，所有模型均为"input_ids.1"。
* output_name：pt格式模型文件输出节点的名称，在执行推理时会打印该信息。因此如果实现不明确，可以先随便填写一个值，然后进行推理，获取准确信息后再进行调整。

* vocab_size：模型词表大小
* model_file：pt格式模型文件所在路径

* serialize_file：序列化文件保留路径，如果该字段内容为""，则不生成序列化文件。
* model_precision: 表示推理时使用的数据类型，当前支持a_bf16_w_int8_kv_int8、a_bf16_w_int8、bf16三种类型。
* build_seq：模型启动后，支持的最大上下文长度
* build_batch：模型启动后，支持的最大推理batchsize

* distribute_param：分布式策略，List中三个元素分别表示TP数目、PP数目、DP数目，当前PP数目和DP数目只支持1
* devices：使用的GPU对应下标序列

* all_models：由多个key-value对组成。如果是离线测试，只需一个key-value对，key中的host_ip和port可以随意填写一个ip和端口号；如果是在线测试，则key-value对数目为所有服务的数目，key中的host_ip和port需要为每个服务实际监听的IP和端口，value为服务名称，需要和model字段一致。

> 注意：在创建配置时，不要增删字段，字段对应值的类型与样例保持一致。

### 采样参数

无论是离线推理还在在线推理，都需要配置采样参数。下表列举了支持的采样参数，和其对应的取值范围。

|采样参数|取值范围|功能介绍|
|-------|-------|--------|
|n | 大于等于1的整数| 单个prompt对应输出语句的数量，默认值为1，如果值过大会导致引擎处理的序列数目超过build_batch，从而报错。|
|presence_penalty|任意浮点数|默认值为0.0。如果值大于0，则鼓励生成未生成过的token；如果值小于0，则鼓励生成已生成的token。|
|frequency_penalty|任意浮点数|默认值为0.0。如果值大于0，则鼓励生成未生成过的token；如果值小于0，则鼓励生成已生成的token。|
|repetition_penalty|任意浮点数|默认值为1.0。如果值大于1.0，则鼓励生成未出现过(包括prompt中的tokens)的token；如果值小于1.0，则鼓励生成已出现过的token。|
|temperature|大于等于0的浮点数|温度系数，用来控制文本生成的随机性和创造性。如果取值为0，则使用greed search来生成token。|
|top_p|(0, 1]|通过累积概率阈值来控制候选token集，默认值为1。|
|top_k|-1或者1~vocab_size之间的整数|指定候选token范围，默认值为-1，即候选集为全部token。|
|stop|List[str]|用于指定终止字符串。|
|stop_tokens_ids|List[int]|用于指定终止tokens。|
|include_stop_str_in_output|True或者False|用于指示是否在生成的文本中添加终止字符串，默认为False。|
|ignore_eos|True或者False|用于指示遇到EOS token是否继续生成token，默认为False。|
|max_tokens|大于等于1的整数|指定最大输出tokens数目，如果max_tokens超过了build_seq的一半，会进行截断。|

以上参数更详细的介绍请参考[vLLM Sampling Parameters](https://docs.vllm.ai/en/stable/dev/sampling_params.html)。

### 推理程序

离线推理使用的参考程序可参考llm_infer.py

### 注意事项

* 如果推理模型时使用了多张卡，并且存在两张卡之间是通过PCIE互联，则需要增加"BR_UMD_DEBUG_P2P_ACCESS_CHECK=0"环境变量。
* br-vllm-serving后端推理引擎为suinfer-llm，更多limitation请参考suinfer-llm使用文档。

## 在线推理

在线服务部署也需要构建模型配置文件，具体步骤与离线推理一致。

### 单服务部署

#### 启动服务

启动服务的命令为：

  ```bash
  # qwen2_72b.json：为模型配置文件
  # port：服务监听端口
  br_api_server --config qwen2_72b.json --port port_id
  ```  

#### 发送请求

* Chat API

  chat api会对原始输入的prompt增加对话模板，然后再送给suinfer-llm引擎进行推理，对应的请求命令格式如下：

  ```bash
  curl http://host_ip:port/v1/chat/completions \
  -H  "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen2-72B",
        "messages": [{
            "role": "user",
            "content": "描述一下夏天"
        }],
        "max_tokens": 70,
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 10,
        "stream": true
      }'
  ```

  注意：

  \- 请求的port要和启动服务时使用的一致，host_ip为服务所在主机IP
  
  \- model字段的值需要和启动服务时使用的qwen2-72b.json中用到的一致

* Completions API

  completions api将原始输入直接送给suinfer-llm引擎进行推理，对应的请求命令格式如下：

  ```bash
  curl http://host_ip:port/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen/Qwen2-72B",
      "prompt":"描述一下夏天",
      "max_tokens": 70,
      "top_p":0.9,
      "top_k":10,
      "temperature": 0.9,
      "stream": true
      }'
  ```

### 多服务部署

以启动qwen1.5_14b和codellama34b两个服务为例。

#### 启动qwen1.5_14b服务

创建模型配置文件qwen1.5_14b.json，注意all_models字段需要把所有服务监听的地址加入进来。
**qwen1.5_14b.json**

```json
{
  "model": "Qwen/Qwen1.5-14B-Chat",
  "input_name": "input_ids.1",
  "output_name": "10",
  "vocab_size": 152064,
  "model_file": "/models/qwen1.5-14b-chat/qwen1.5-14b-chat.pt",
  "serialize_file": "/root/qwen1.5_14b_chat",
  "model_precision": "a_bf16_w_int8_kv_int8",
  "build_seq": 2048,
  "build_batch": 12,
  "distribute_param": [2, 1, 1],
  "devices": [6, 7],
  
  "all_models": {
    "http://host_ip1:port1": "Qwen/Qwen1.5-14B-Chat",
    "http://host_ip2:port2": "codellama/CodeLlama-34b-hf"
  }
}
```

启动qwen1.5_14b服务，启动命令为：

```bash
# host和port需要是实际监听的地址
br_api_server --config_path qwen1.5_14b.json --port port1
```

#### 启动codellama34b服务

创建模型配置文件codellama34b.json，注意all_models字段需要把所有服务监听的地址加入进来。
**codellama34b.json**

```json
{
  "model": "codellama/CodeLlama-34b-hf",
  "input_name": "input_ids.1",
  "output_name": "10",
  "vocab_size": 32000,
  "model_file": "/models/codellama34b/codellama34b.pt",
  "serialize_file": "/root/codellama34b",
  "model_precision": "a_bf16_w_int8_kv_int8",
  "build_seq": 2048,
  "build_batch": 16,
  "distribute_param": [2, 1, 1],
  "devices": [2, 3],
  
  "all_models": {
    "http://host_ip1:port1": "Qwen/Qwen1.5-14B-Chat",
    "http://host_ip2:port2": "codellama/CodeLlama-34b-hf"
  }
}
```

启动codellama34b服务，启动命令为：

```bash
# host和port需要是实际监听的地址
br_api_server --config_path codellama34b.json --port port2
```

#### 启动nginx

首先需要安装nginx，nginx可以安装在以上两个服务所在的容器中，也可以单独启动一个容器，将nginx安装在其中。安装完成后，设置nginx配置文件，编辑/etc/nginx/sites-enabled/brvllm.conf文件：
**brvllm.conf**

```conf
# port3为nginx监听端口
upstream brvllm_server {
    server host_ip1:port1;
    server host_ip2:port2;
}
 
server {
    listen port3;
     
    location /v1 {
        proxy_pass http://brvllm_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

启动nginx，命令如下：

```bash
nginx -t
service nginx start
```

#### 发送请求

请求的格式与单服务部署时一致，只需将其中的host_ip和port修改为nginx监听的地址即可。

### Metrics查询

查询参考脚本为：

```python
import requests
# host_ip和port为服务监听端口
response = requests.get("http://host_ip:port/metrics")
print(response.text)
```

执行上述脚本，即可获取vLLM 0.4.0.post1定义的Metrics信息。

<div style="page-break-after:always"></div>

## 已知限制

* 如果推理模型时只使用了单张卡，则只能使用0号卡；
* 最大上下文长度不能超过8K；
* 使用br-vllm-serving时，不能开启graph capture；
* 模型配置文件中build_batch字段的值建议小于等于64，否则会报错。
* 执行推理时需要加上BR_UMD_DEBUG_P2P_ACCESS_CHECK=0环境变量
* 使用在线推理时，如果遇到“Error in applying chat template from request”信息，表示只能使用Completions API。
* 对于仓库中提供的大模型样例，推荐使用如下配置

  |模型|推荐TP策略|推荐推理精度|
  |-------|-------|--------|
  |LLaMa2 7B|TP1|a_bf16_w_int8_kv_int8|
  |Baichuan2 7B|TP1|a_bf16_w_int8_kv_int8|
  |ChatGLM3 6B|TP1|a_bf16_w_int8_kv_int8|
  |Mistral 7B|TP1|a_bf16_w_int8_kv_int8|
  |Qwen1.5 14B|TP2|a_bf16_w_int8_kv_int8|
  |LLaMa2 70B|TP4/TP8|a_bf16_w_int8_kv_int8|
  |Qwen2 72B|TP4/TP8|a_bf16_w_int8_kv_int8|


## 法律声明

**著作权©**

壁仞科技2020-2024，版权所有。未经壁仞科技事先书面许可，本文档内容不得以任何形式将其复制、修改、出版、传输或发布。

**商标。**

本文档所包含的任何壁仞科技的商号、商标、图形标志和域名，均为壁仞科技所有。未经壁仞科技事先书面许可，不得以任何形式将其复制、修改、出版、传输或发布。

**性能信息**。

本文档中所包含的性能指标包括设计规格、模拟测试指标以及特定环境下的测试和评估指标。设计规格为产品设计时拟定的指标，仅用于提供信息的目的而供您参考，实测指标将以具体的测试数据为准。模拟测试指标是通过在体系结构模拟器上运行模拟而获得，仅用于提供信息目的。该类测试的系统硬件、软件设计或配置的任何不同都可能影响实际性能。特定环境下的测试和评估指标系采用特定的计算机系统或组件操作而获得，可反映出我司产品的大致性能。系统硬件、软件设计或配置的任何不同都可能影响实际性能。

**前瞻性陈述。**

本文档的信息可能包含前瞻性陈述，可能存在风险和不确定性。请勿仅依赖于上述信息做出您的商业决定。

**注意。**

本产品后续可能进行版本升级，本文档内容会不定期更新。除非在合同中另有约定，本文档仅作产品使用指导，其中的信息和建议不构成任何明示或暗示的担保。
