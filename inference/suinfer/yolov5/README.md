# suInfer
# YOLOv5 Suinfer离线推理

此链接提供YOLOv5 ONNX模型在BR硬件上离线推理的脚本和方法
在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
| suInfer | >=2.0 |
| biren-pysuinfer | >=1.4 |
| driver |02_6361 |
| 芯片平台| BR10X/BR110 |

## 快速指南

### 1. 安装环境依赖
```
pip3 install -r ./requirements.txt
```


### 2. 下载数据集和预处理

1. 数据集使用coco2017测试数据集，下载链接：[url](http://images.cocodataset.org/zips/val2017.zip)，下载数据放在当前目录./val2017中。

2. 使用yolov5_data_preprocess.py对数据集进行预处理。    
 ```
 python3 yolov5_data_preprocess.py --src ./val2017 --dst ./val2017_processed_fp32 --data_map_path ./data_maps
 ```

前处理后的目录结构:
```
yolov5/
├── README.md
├── cal_yolov5_acc.py	# 后处理脚本
├── data_maps	# 前处理生成的后处理参数
│   ├── map_val.txt
│   └── shape_map.json
├── postprecess_para	# 后处理参数
│   ├── coco_classes.txt
│   └── yolo_anchors.txt
├── pysuinfer_demo.py	# 推理运行脚本
├── requirements.txt	# python依赖
├── val2017	# 下载的数据集目录
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   ├── 000000000632.jpg
│   ├── .....
│   ├── 000000581482.jpg
│   ├── 000000581615.jpg
│   └── 000000581781.jpg
├── val2017_processed_fp32	# 前处理后的数据集目录
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   ├── 000000000632.jpg
│   ├── .....
│   ├── 000000581482.jpg
│   ├── 000000581615.jpg
│   └── 000000581781.jpg
└── yolov5_data_preprocess.py	# 前处理脚本
```



### 3. 离线推理

**离线模型转换**

1. Pytorch模型转换为onnx模型
参考此URL[url](https://docs.ultralytics.com/yolov5/tutorials/model_export/#colab-pro-cpu)转换onnx文件。<br />Pytorch模型转换为onnx后放在当前目录下的./model中。
  

**离线模型推理**
- 程序运行平台
 BIRENSUPA软件平台    

- 开始运行:

  ```
  python3 pysuinfer_demo.py --batchsize 1 --model_path ./model/yolov5s.onnx --ori_data ./val2017_processed_fp32 --logs_path ./yolov5/ --out_path ./yolov5_out --serialize ./serialize
  ```
 详细参数及说明可以使用python3 pysuinfer_demo.py -h打印参考。

- 后处理文件参数(instances_val2017.json)下载：[url](https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json)。下载后放在./postprecess_para目录下。

- 运行后处理脚本:
  ```
  python3 cal_yolov5_acc.py --bin_path ./yolov5_out --cocoGt_path ./postprecess_para/instances_val2017.json --val_map_file_path ./data_maps/map_val.txt  --shape_map_file_path ./data_maps/shape_map.json --origin_images_path ./val2017 --processed_images_path ./val2017_processed_fp32 --coco_class_path ./postprecess_para/coco_classes.txt --yolo_anchors_path ./postprecess_para/yolo_anchors.txt
  ```    
  详细参数及说明通过python3 cal_yolov5_acc.py -h查看

  
## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**  |   Map 0.5   |
| :---------------: | :-------: | :-----------: |
| offline Inference | 5000 images | 0.552 |
  

## 参考
[1] https://github.com/ultralytics/yolov5
