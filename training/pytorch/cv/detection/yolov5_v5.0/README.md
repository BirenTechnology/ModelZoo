# YOLOv5m_v5.0_IDff0d3778
# 概述

## 简述

YOLO是一个经典的物体检测网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。
基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov5/tree/v5.0
  commit_id=f5b8f7d54c9fa69210da0177fec7ac2d9e4a627c
  ```

- 适配壁仞 AI 处理器的实现：

  ```
  url=https://github.com/BirenTechnology/ModelZoo
  code_path=training/pytorch/cv/detection
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。



# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、pytorch以及br_pytorch 如下表所示。 

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | Br10X |
  | 固件与驱动  | master_3370 |
  | pytorch    | 1.10.0+cpu |
  | br_pytorch | 1.10.0+bde8c64 |


- 环境准备指导。 

  安装对应版本固件、驱动、pytorch以及br_pytorch

- 安装依赖。

    Python 3.8 环境下

  ```
  bash ./requirements.sh
  ```
  
## 准备数据集


   模型启动自动下载数据集，手动下载请参考: data/scripts/get_coco.sh


# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./run.sh

     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./dist_train.sh 

     ```

   模型训练脚本参数说明如下。

      ```
      公共参数：
      --device                            //训练指定训练用卡
      --data                              //训练所需的yaml文件
      --cfg                               //训练过程中涉及的参数配置文件
      --weights                           //权重
      --batch-size                        //训练批次大小
      --workers                           //dataloader线程数
      ```
   
      训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME     | mAP_0.5    | FPS    | Epochs | Torch_version |
|--------  | ------ |:-------| ------ | :------------ |
| 8p | 0.625 | 683.5.1 | 300 | 1.11 |

# 版本说明

# 公网地址说明

## 变更

2024.07.19：更新Readme发布。

## 已知问题

无。
