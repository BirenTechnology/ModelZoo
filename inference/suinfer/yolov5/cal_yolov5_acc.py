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


import argparse
import json
import os
import time
import shutil

import numpy as np
import torch
import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
sys.path.append("../object_detect_utils/")

from utils import get_anchors, get_classes
from utils_bbox import DecodeBox


class YoloAccu:
  def __init__(self, args) -> None:
    self.predicted_file_path = args.predicted_file_path
    # self.predicted_path, _ = os.path.split(self.predicted_file_path)
    self.predicted_path = '/tmp'
    self.val_map_file_path = args.val_map_file_path
    self.shape_map_file_path = args.shape_map_file_path
    self.cocoGt_path = args.cocoGt_path
    self.coco_class_path = args.coco_class_path
    self.anchors_path = args.yolo_anchors_path
    self.origin_images_path = args.origin_images_path
    self.draw_bbox = args.draw_bbox
    self.real_class_name = []
    self.to_draw = True if args.draw_bbox == "YES" else False

    self.val_map = self._get_map_val()
    self.shape_map = self._get_shape_map()

    self.buffer_type = args.buffer_type
    self.bin_path = args.bin_path

    self.confidence = args.confidence
    self.nms_iou = args.nms_iou
    self.input_shape = [640, 640]
    self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    self.class_names, self.num_classes = get_classes(self.coco_class_path)
    self.anchors, self.num_anchors = get_anchors(self.anchors_path)
    self.bbox_util = DecodeBox(self.anchors, self.num_classes,
                               (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
    self.letterbox_image = True

    self.cocoGt = COCO(self.cocoGt_path)
    self.img_ids_to_delete = []
    self.res_name = "eval_results.json"

  def draw_bboxes(self, predicted_data):
    root_draw_objs_name = "drawed_objs"
    root_draw_objs_fold_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(self.bin_path))), root_draw_objs_name)
    if os.path.exists(root_draw_objs_fold_path):
      shutil.rmtree(root_draw_objs_fold_path)
    os.makedirs(root_draw_objs_fold_path)

    filtered_predicted_data = [bbox for bbox in predicted_data if bbox['score'] >= 0.5]
    drawed = {}
    for bbox in filtered_predicted_data:
      if bbox['image_id'] not in drawed.keys():
        origin_img_path = os.path.join(self.origin_images_path, str(bbox['image_id']).zfill(12) + ".jpg")
        # print(origin_img_path)
        drawed[bbox['image_id']] = cv2.imread(origin_img_path)

      top_left = (int((bbox['bbox'][0]) ), int(( bbox['bbox'][1]) ))
      bottom_right = (int( top_left[0] + bbox['bbox'][2] ), int(top_left[1] + bbox['bbox'][3] ))

      color = (0, 0, 255)  # red
      thickness = 2
      cv2.rectangle(drawed[bbox['image_id']], top_left, bottom_right, color, thickness)
      class_name = self.class_names[self.real_class_name.index(bbox['category_id'])]
      cv2.putText(drawed[bbox['image_id']], class_name, top_left, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

    for k, v in drawed.items():
      cv2.imwrite(os.path.join(root_draw_objs_fold_path, str(k).zfill(12) + ".jpg"), v)

    print("===============================================================================")
    print("You can check objects in " + root_draw_objs_fold_path)
    print("===============================================================================")

  def cal_yolo_accu(self):
    predicted_data = self._get_predicted()
    predicted_data = self._parser_predicted_data(predicted_data)
    #for bbox in predicted_data:
    #  print(bbox)
    self._cal_accu()
    if self.to_draw:
      self.draw_bboxes(predicted_data)

  def _get_predicted(self):
    predicted_res = {} # name : data
    if self.buffer_type == "json":
      with open(self.predicted_file_path, 'r') as accuf:
        for line in accuf.readlines():
          if line.strip() == '[' or line.strip() == ']' or line.strip() == '':
            continue
          line_good = line.strip()
          if line.strip()[-1] == ",":
            line_good = line.strip()[:-1]
          d = json.loads(line_good)
          qsl_idx = int(d["qsl_idx"])
          image_name = self.val_map[qsl_idx].strip()
          data = np.frombuffer(bytes.fromhex(d["data"]), np.float32)
          predicted_res[image_name] = data
    elif self.buffer_type == "bin":
      bin_list = os.listdir(self.bin_path)
      for bin_file in bin_list:
        # remove '.out' of 'image_name.out'
        image_name = '.'.join(bin_file.split('.')[:-1])
        # label = label = imagenet_dict[image_name]
        data = np.fromfile(self.bin_path + '/' + bin_file, np.float32)
        predicted_res[image_name] = data
    else:
      assert False, "unsupport buffer type"
    
    image_id_exist = self.cocoGt.getImgIds()

    predicted_res_ids = []
    for image_name in predicted_res.keys():
      image_id = int(image_name.strip()[:-4])
      predicted_res_ids.append(image_id)

    self.img_ids_to_delete = [id for id in image_id_exist if id not in predicted_res_ids]

    return predicted_res

  def _get_shape_map(self):
    """
    get a mapping from filename to image shape
    """
    with open(self.shape_map_file_path, 'r') as f:
      return json.load(f)

  def _get_map_val(self):
    """
    get a mapping from sql_idx to filename
    """
    res = []
    with open(self.val_map_file_path, 'r') as map_val:
      lines = map_val.readlines()
      for line in lines:
        res.append(line)
    return res

  def _cal_accu(self):

    for imgId in self.img_ids_to_delete:
      self.cocoGt.imgs.pop(imgId)
    cocoDt = self.cocoGt.loadRes(os.path.join(
        self.predicted_path, self.res_name))
    cocoEval = COCOeval(self.cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("Get map done.")

  def _parser_predicted_data(self, predicted_datas):
    clsid2catid = self.cocoGt.getCatIds()
    self.real_class_name = clsid2catid
    results = []
    for image_name in predicted_datas.keys():
      predicted_data = predicted_datas[image_name].copy()

      bbox_output = predicted_data.reshape(-1, 85)
      # if bbox_output.shape[0] > 0:
      #   print(bbox_output[-1])

      bbox_output[:, 0:4] = bbox_output[:, 0:4] / float(self.input_shape[0])
      bbox_output = torch.from_numpy(bbox_output).unsqueeze(0)

      origin_shape = np.array(
          self.shape_map[image_name]["origin_shape"])
      output = self.bbox_util.non_max_suppression(bbox_output, self.num_classes, self.input_shape,
                                                  origin_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)

      output = output[0]
      if output is None:
          continue
      top_label = np.array(output[:, 6], dtype='int32')
      top_conf = output[:, 4] * output[:, 5]
      top_boxes = output[:, :4]

      for i, c in enumerate(top_label):
          result = {}
          top, left, bottom, right = top_boxes[i]

          result["image_id"] = int(image_name.strip()[:-4])
          result["category_id"] = clsid2catid[c]
          result["bbox"] = [float(left), float(
              top), float(right-left), float(bottom-top)]
          result["score"] = float(top_conf[i])
          results.append(result)

    self.res_name = str(time.time()) + "_" + self.res_name
    with open(os.path.join(self.predicted_path, self.res_name), "w") as f:
      json.dump(results, f)

    return results


def get_params():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--predicted_file_path", default="./mlperf_log_accuracy.json")
  parser.add_argument(
      "--cocoGt_path", default="./postprecess_para/instances_val2017.json")
  parser.add_argument(
      "--val_map_file_path", default="./data_maps/map_val.txt")
  parser.add_argument(
      "--shape_map_file_path", default="./data_maps/shape_map.json")
  parser.add_argument(
      "--origin_images_path", default="./val2017")
  parser.add_argument(
      "--processed_images_path", default="./val2017_processed_fp32")
  parser.add_argument(
      "--coco_class_path", default="./postprecess_para/coco_classes.txt")
  parser.add_argument(
      "--yolo_anchors_path", default="./postprecess_para/yolo_anchors.txt")
  parser.add_argument("--buffer_type", default="bin", choices=["json", "bin"], help="file type of predicted data")
  parser.add_argument("--bin_path", default="./yolov5_out", help="path to result.bin")
  parser.add_argument("--confidence", type=float, default=0.001, help="threshold")
  parser.add_argument("--nms_iou", type=float, default=0.5, help="path to result.bin")
  parser.add_argument("--draw_bbox", default="NO")
  args = parser.parse_args()
  return args


def main():
  args = get_params()
  yolo_accu = YoloAccu(args)
  yolo_accu.cal_yolo_accu()


if __name__ == "__main__":
  main()
