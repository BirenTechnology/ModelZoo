# YOLOv5m_v6.0_ID9057e273

[toc]

# 概述

## 模型介绍

Yolov5算法是目前应用最广泛的目标检测算法之一，它基于深度学习技术，在卷积神经网络的基础上加入了特征金字塔网络和SPP结构等模块，从而实现了高精度和快速检测速度的平衡。

## 支持任务列表

本仓已经支持以下模型任务类型

|      模型      | 任务列表 | 是否支持 |
| :------------: | :------: | :------: |
|    Yolov5m     |  预训练  |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov5/tree/v6.0 

  commit_id=956be8e6
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

   -v <path_to_br_pytorch_model_zoo>:/workspace/br_pytorch_model_zoo birensupa-pytorch:<version> /bin/bash

   ```

   | 参数                                                    | 描述                                                         |
   | ------------------------------------------------------- | ------------------------------------------------------------ |
   | --shm-size='256g'                                       | 设置 shm size 为 256g。                                      |
   | --network=host                                          | （可选）设置网络模式为 host 网络。                           |
   | --device /dev/biren:/dev/biren                          | 挂载 biren 设备。                                            |
   | -v <path_to_parent_path_coco2017>:/workspace/datasets         | 挂载数据集。                                 |
   | -v <host_kernel_cache_parent_path>:/workspace/model_kernel_cache          | 挂载kernel_cache。                                 |
   | -v <path_to_br_pytorch_model_zoo>:/workspace/br_pytorch_model_zoo | 挂载 br_pytorch_model_zoo 目录。                             |

### Docker 端操作

1. 安装环境依赖 。

  - 安装环境。

    ```bash
    cd /workspace/br_pytorch_model_zoo/cv/ultralytics-yolov5m
    pip3 install -r requirements.txt
    pip3 uninstall wandb
    pip3 install Pillow==9.5
    ```


## 开始训练

1. 训练准备。

   ```bash
   cd /workspace/br_pytorch_model_zoo/cv/ultralytics-yolov5m

   # 拷贝kernel_cache到对应目录下并修改脚本中名称
   cp -r /workspace/model_kernel_cache/kernel_cache_yolov5m /root
   sed -i 's/v5m-kernel-cache/kernel_cache_yolov5m/g'  dist_train.sh  

   # link数据集到对应目录
   ln -snf /workspace/datasets/coco2017  /workspace/br_pytorch_model_zoo/cv/coco   
   ```
2. 执行训练。

   ```bash
   bash dist_train.sh 2>&1 |tee rel_2411_Yolov5m.log
   ```


   > **说明：**
   > 由于训练时间较长，建议使用[tmux](https://github.com/tmux/tmux/wiki)等工具后台执行，避免控制台中断。
   >
4. 查看 log 和 tensorboard

   ```bash
   # log 被重定向到了 rel_2411_Yolov5m.log 文件下

   # tensorboard 和其他数据存在放runs目录下
   ```

### 可改参数

修改 `/workspace/br_pytorch_model_zoo/cv/ultralytics-yolov5/dist_train.sh`  脚本里的参数

| 参数 | 描述 |
| --- | --- |
|data|指定使用的数据集|
|cfg|指定运行的yolov5配置，本文为Yolov5m|
|epochs|训练的epoch数|
|device|指定使用的GPU|

> 注意！以上参数不宜建议修改，否则精度性能数据无法保证。参数详细含义参考 /workspace/br_pytorch_model_zoo/cv/ultralytics-yolov5m/train.py。

## 训练结果展示

**机器配置：**

|  NAME | 配置
| ---- | -----
|  GPU版本类型和型号 | Biren106M
|  CPU型号/核数/主频| Intel(R) Xeon(R) Platinum 8462Y+
| 硬盘类型及容量 | GPFS
| 内存根数及大小 | 2T=64G*32
| OS和内核版本 |Linux version 5.4.0-125-generic Ubuntu 20.04.5 LTS
| Flash FW版本 | 001050100091
| 网卡 | ROCE_v2 100G

### 性能

Throughput 计算方法：

global_batch_size  / step_time

-`global_batch_size`：总 batch_size，本文中单卡32，8卡为 32 * 8 = 256；

-`step_time`：一个step花费的时间，为训练打印数据的倒数；

计算方式示例：
选取某个epoch打印的性能数据，如 2.80it/s，Throughput = 256 / (1 / 2.80) = 716 samples/s


**表 4** 训练结果展示表，仅供参考

|  NAME | 集群 |TGS
| ---- | ----- | ------
| Yolov5m -BR10X | 1x8 | 716
| Yolov5m -参考| 1x8 | -

### Loss

Yolov5m，基于1机8卡，实测100个epoch， loss稳定下降。

### 精度

vim rel_2411_yolov5m.log，查看最后一个epoch打印的精度数据，如：mAP50：0.669；mAP50-95：0.503



# 版本说明

当前版本号：v0.1

## 变更

|  版本号 | 时间  | 概况  |
| ------------ | ------------ | ------------ |
|  v0.1 |  2024.11.19 |  首次发布，支持 Yolov5m 预训练  |

## FAQ

无。
