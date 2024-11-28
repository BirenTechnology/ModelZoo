# Yolov8l/x_IDec7dc353bc8

[toc]

# 概述

## 模型介绍

Yolov8 系列模型建立在以前 YOLO 版本的成功基础上，并引入了新功能和改进，以进一步提高性能和灵活性。YOLOv8 旨在快速、准确且易于使用，使其成为各种对象检测和跟踪、实例分割、图像分类和姿势估计任务的绝佳选择

## 支持任务列表

本仓已经支持以下模型任务类型

|      模型      | 任务列表 | 是否支持 |
| :------------: | :------: | :------: |
|    Yolov8x     |  预训练  |    ✔     |
|    Yolov8l     |  预训练  |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/ultralytics/ultralytics/tree/v8.2.45

  commit_id=69cfc8aa
  ```

# 环境依赖

  **表 1**  系统支持表

|   系统   | 支持版本 |
| :---------: | :------: |
|   Ubuntu   |  22.04   |

  **表 2**  三方库版本支持表

|   三方库    | 支持版本 |
| :---------: | :------: |
|   PyTorch   |  1.12.1   |


# 预训练

## 准备环境

<table><tr><tdbgcolor=#ffeccc><b>说明：</b><ul><li>您可以联系壁仞产品服务团队获取软件包和源码。</li><li>本文中提到的软件安装包或镜像文件名称仅为示例。实际操作时，请以当时获取的软件版本和文件名称为准。</li></ul></td></tr></table>

### Host 端操作

**注意： `<version>` 表示软件包版本号，请根据实际获取的软件版本进行替换。**

1. 准备数据集： 自行准备 coco2017 数据集。下载地址：https://cocodataset.org/#home


2. 安装驱动。

   ```bash
   sudo bash biren-driver_<version>_linux-x86_64.run
   ```
3. 安装 biren-container-toolkit。

   ```bash
   sudo bash biren-container-toolkit_<version>_linux-x86_64.run
   ```
4. 获取基础镜像。

   ```bash
   docker load -i birensupa-pytorch-<version>.tar
   ```
5. 启动容器。

   ```bash
   docker run --name <container_name> -it -d \

   --shm-size='256g'\

   --network=host  \

   --device/dev/biren:/dev/biren \

   -v <path_to_parent_path_coco2017>:/workspace/datasets \

   -v <host_kernel_cache_parent_path>:/workspace/model_kernel_cache \

   -v br_pytorch_model_zoo:/workspace/br_pytorch_model_zoo birensupa-pytorch:<version> /bin/bash
   ```

   | 参数                                                    | 描述                                                         |
   | ------------------------------------------------------- | ------------------------------------------------------------ |
   | --shm-size='256g'                                       | 设置 shm size 为 256g。                                      |
   | --network=host                                          | （可选）设置网络模式为 host 网络。                           |
   | --device /dev/biren:/dev/biren                          | 挂载 biren 设备。                                            |
   | -v <path_to_parent_path_coco2017>:<docker_datasets_path>          | 挂载数据集。                                 |
   | -v <host_kernel_cache_parent_path>:/workspace/model_kernel_cache          | 挂载kernel_cache。                                 |
   | -v  br_pytorch_model_zoo:/workspace/br_pytorch_model_zoo | 挂载 br_pytorch_model_zoo 目录。                             |

### Docker 端操作

1. 安装环境依赖 。

  - 安装环境。

    ```bash
    cd /workspace/br_pytorch_model_zoo/cv/ultralytics
    python3 -m pip install -e .
    export PYTHONPATH=/workspace/br_pytorch_model_zoo/:$PYTHONPATH
    ```

## 开始训练

1. 训练准备。

   ```bash

   cd /workspace/br_pytorch_model_zoo/cv/ultralytics

   # 拷贝yolov8x的kernel_cache到对应目录下并修改脚本中名称
   cp -r /workspace/model_kernel_cache/kernel_cache_yolov8x /root
   sed -i 's/yolo_kernel_cache/kernel_cache_yolov8x/g' run_yolov8x.sh

   # 拷贝yolov8l的kernel_cache到对应目录下并修改脚本中名称
   cp -r /workspace/model_kernel_cache/kernel_cache_yolov8l /root
   sed -i 's/yolo_kernel_cache/kernel_cache_yolov8l/g' run.sh

   # link数据集到对应目录
   mkdir -p datasets && ln -snf /workspace/datasets/coco2017 ./datasets/coco  

   # 准备权重文件 yolov8n.pt 和字体文件 Arial.ttf
   # 方式一：有网环境下，启动训练后会自动下载
   # 方式二：无网环境下，分别将如下两个文件拷贝到容器内对应位置：
   yolov8n.pt：用于amp前置验证。拷贝到容器内 /workspace/br_pytorch_model_zoo/cv/ultralytics 目录下
   Arial.ttf：用于设置字体。拷贝到容器内 /root/.config/Ultralytics 目录下
   ```
2. 执行训练。

   ```bash
   # Yolov8x 训练
   bash run_yolov8x.sh 2>&1 |tee rel_2411_yolov8x.log

   # Yolov8l 训练
   bash run.sh 2>&1 |tee rel_2411_yolov8l.log   
   ```


   > **说明：**
   > 由于训练时间较长，建议使用[tmux](https://github.com/tmux/tmux/wiki)等工具后台执行，避免控制台中断。
   >
4. 查看 log 和 tensorboard

   ```bash
   # log 被重定向到了 rel_2411_yolov8x.log 文件下
   # tensorboard 数据在 /workspace/br_pytorch_model_zoo/cv/ultralytics/runs 目录下生成，可使用tensorboard打开
   ```

### 可改参数（以yolov8x为例）

修改 `/workspace/br_pytorch_model_zoo/cv/ultralytics/run_yolov8x.sh`  脚本里的参数

| 参数 | 描述 |
| --- | --- |
|data|指定使用的数据集|
|model|指定运行的yolov8版本|
|epochs|训练的epoch数|
|imgsz|图片分辨率|
|device|指定使用的GPU|

> 注意！以上参数不建议修改，否则精度性能数据无法保证。

## 训练结果展示

**机器配置：**

|  NAME | 配置
| ---- | -----
|  GPU版本类型和型号 | Biren106M
|  CPU型号/核数/主频| Intel(R) Xeon(R) Platinum 8462Y+
| 硬盘类型及容量 | GPFS
| 内存根数及大小 | 2T=64G*32
| OS和内核版本 |Linux version 5.4.0-125-generic Ubuntu 22.04 LTS
| Flash FW版本 | 001050100091
| 网卡 | ROCE_v2 100G

### 性能

Throughput 计算方法：

global_batch_size  / step_time

-`global_batch_size`：总 batch_size，本文中单卡16，8卡为 16 * 8 = 128；

-`step_time`：一个step花费的时间，等于训练过程中打印数据的倒数；

计算方式示例：
选取某个epoch打印的性能数据，如:
yolov8x： 1.53 it/s，Throughput = 128 / (1 / 1.53) = 195 samples/s
yolov8l： 1.75 it/s，Throughput = 128 / (1 / 1.75) = 224 samples/s


**表 4** 训练结果展示表，仅供参考

|  NAME | 集群 |Throughput
| ---- | ----- | ------
| Yolov8x -BR10X | 1x8 | 195
| Yolov8x -参考| 1x8 | 666
| Yolov8l -BR10X | 1x8 | 224
| Yolov8x -参考| 1x8 | -

### Loss

Yolov8x，基于1机8卡，实测100个epoch， loss稳定下降。
Yolov8l，基于1机8卡，实测100个epoch， loss稳定下降。

### 精度

- yolov8x：
  vim rel_2411_yolov8x.log，查看最后一个epoch打印的精度数据，如：mAP50：0.669；mAP50-95：0.503
- yolov8l：
  vim rel_2411_yolov8l.log，查看最后一个epoch打印的精度数据，如：mAP50：0.679；mAP50-95：0.512



# 版本说明

当前版本号：v0.1

## 变更

|  版本号 | 时间  | 概况  |
| ------------ | ------------ | ------------ |
|  v0.1 |  2024.11.19 |  首次发布，支持 Yolov8l/x 预训练  |

## FAQ

无。
