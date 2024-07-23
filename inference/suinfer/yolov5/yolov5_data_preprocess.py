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


from threading import Lock
import functools
import json
import os
import subprocess
import argparse
# import copy

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
# from utils.data_convert import *

src_default_path = "./val2017"
dst_default_path = "./val2017_processed_fp32"
# src_default_path = ""
# dst_default_path = ""

lock = Lock()
image_shape_dict = {}

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#


def resize_image(image, size, letterbox_image):
  iw, ih = image.size
  w, h = size
  if letterbox_image:
      scale = min(w/iw, h/ih)
      nw = int(iw*scale)
      nh = int(ih*scale)

      image = image.resize((nw, nh), Image.BICUBIC)
      new_image = Image.new('RGB', size, (128, 128, 128))
      new_image.paste(image, ((w-nw)//2, (h-nh)//2))
  else:
      new_image = image.resize((w, h), Image.BICUBIC)
  return new_image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def pad(image):
    image = torch.from_numpy(image)
    image = torch.nn.functional.pad(image, (0, 0, 0, 0, 0, 5))
    image = image.numpy()
    return image


def preprocess(image):
    input_shape = [640, 640, 3]
    #---------------------------------------------------#
    #   计算输入图片的高和宽
    #---------------------------------------------------#
    image_shape = np.array(np.shape(image)[0:2])
    #---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #---------------------------------------------------------#
    image = cvtColor(image)
    #---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    #---------------------------------------------------------#
    image_data = resize_image(image, (input_shape[1], input_shape[0]), True)
    #---------------------------------------------------------#
    #   添加上batch_size维度
    #---------------------------------------------------------#
    image_data = np.transpose(
        np.array(image_data, dtype='float32')/255.0, (2, 0, 1))

    # image_data = pad(image_data)

    # image_data = fp32Tobf16(image_data)

    # tensor_n_cp = copy.deepcopy(image_data)

    # image_data = convertConvActivation(torch.from_numpy(
    #     tensor_n_cp).reshape(1, *tensor_n_cp.shape))

    return image_data, image_shape


def generate_map(src_path, data_map_path):

    def cmp_fun(str1, str2):
        if int(str1[:-4]) < int(str2[:-4]):
            return -1
        elif int(str1[:-4]) > int(str2[:-4]):
            return 1
        else:
            return 0

    sample_list = os.listdir(src_path)
    sample_list.sort(key=functools.cmp_to_key(cmp_fun))
    with open("map_val.txt", 'w') as map_val:
        for sample in sample_list:
            map_val.write(sample + '\n')

    command = "mv map_val.txt " + data_map_path + "/map_val.txt"
    subprocess.call(command, shell=True)


def generate_shape_map(image_shape_dict, data_map_path):

    with open("shape_map.json", "w") as f:
        json.dump(image_shape_dict, f)

    command = "mv shape_map.json " + data_map_path + "/shape_map.json"
    subprocess.call(command, shell=True)


def do_work(image_path):
    image = Image.open(src_default_path + '/' + image_path)
    data, image_shape = preprocess(image)
    data.tofile(dst_default_path + '/' + image_path)


def main():
    global src_default_path
    global dst_default_path
    parser = argparse.ArgumentParser(description="yolov5 data preprocess")
    parser.add_argument(
        "--src",
        type=str,
        metavar="origin images",
        default=src_default_path,
        help="origin image dirs."
    )
    parser.add_argument(
        "--dst",
        type=str,
        metavar="preprocessed images",
        default=dst_default_path,
        help="preprocessed image dirs."
    )
    parser.add_argument(
        "--data_map_path",
        type=str,
        metavar="data_map_path",
        default="./data_maps",
        help="data_map_path."
    )

    args = parser.parse_args()
    src_default_path = args.src
    dst_default_path = args.dst

    assert os.path.exists(src_default_path), "input image path is not exist"
    #   if os.path.exists(dst_default_path):
    #     shutil.rmtree(dst_default_path)
    if not os.path.exists(dst_default_path):
        os.makedirs(dst_default_path, 0o755)

    if not os.path.exists(args.data_map_path):
        os.makedirs(args.data_map_path, 0o755)

    image_list = os.listdir(src_default_path)
    # for image_name in tqdm(image_list, desc='single thread'):
    #     image = Image.open(origin_images_path + '/' + image_name)
    #     data, image_shape = preprocess(image)
    #     data.tofile(processed_image_path + '/' + image_name)
    #     image_shape_dict[image_name] = {"input_shape": [
    #         640, 640], "origin_shape": image_shape.tolist()}

    tuple_args = (image_path for image_path in image_list)
    from multiprocessing import Process, Pool
    cpus = 24
    with Pool(cpus) as pool:
        list(tqdm(pool.imap(do_work, tuple_args), total=len(
            image_list), desc='multi thread({}): '.format(cpus)))

    for image_path in tqdm(image_list, desc='shape map       '):
        image = Image.open(args.src + '/' + image_path)
        image_shape = np.array(np.shape(image)[0:2])
        image_shape_dict[image_path] = {"input_shape": [
            640, 640], "origin_shape": image_shape.tolist()}

    generate_map(args.src, args.data_map_path)
    generate_shape_map(image_shape_dict, args.data_map_path)


if __name__ == "__main__":
    main()
    # image = Image.open('/work/000000000471.jpg')
    # data, image_shape = preprocess(image)
    # data.tofile('/work/yolov5_input_bf16.bin')
