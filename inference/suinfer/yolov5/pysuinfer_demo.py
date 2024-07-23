#!/usr/bin/env python
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
import numpy as np
import cv2 as cv
import argparse
from pysuinfer import *
import sys
import torch
import onnx
import copy
from tqdm import tqdm


THIS_FILE_DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, THIS_FILE_DIR + "/.")

def prepareOutput(session, output_names):
    output = {}
    for out_name in output_names:
        modelinfo = session.GetOutputInformation(out_name)
        output[out_name] = np.ones(modelinfo.dims, np.float32)
    return output


def runSession(session, inputs, outputs):
    for input_name in inputs.keys():
        inputinfo = session.GetInputInformation(input_name)
        if torch.is_tensor(inputs[input_name]):
            inputs[input_name] = inputs[input_name].numpy()
        if inputinfo.dims[0] > inputs[input_name].shape[0]:
            tmp_data = np.zeros(inputinfo.dims, inputs[input_name].dtype)
            for i in range(inputs[input_name].shape[0]):
                tmp_data[i] = inputs[input_name][i]
        else:
            tmp_data = inputs[input_name]
        inputs[input_name] = tmp_data
        session.SetInputInformation(
            input_name, inputs[input_name].view(np.int8), inputinfo
        )

    for out_name in outputs.keys():
        modelinfo = session.GetOutputInformation(out_name)
        session.SetOutputInformation(
            out_name, outputs[out_name].view(np.int8), modelinfo
        )

    if session.Run() != suResult.SUI_STATUS_SUCCESS:
        print("Failed Run session")
        exit()

def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batchsize",
        type=int,
        default=16,
        help="Max device batch size",
    )

    parser.add_argument("--device", default="0", help="device idx")

    parser.add_argument(
        "--model_path",
        default=("./models/yolov5s.onnx"),
    )

    parser.add_argument(
        "--ori_data",
        default=os.path.join("./val2017_processed_fp32"),
    )
    
    parser.add_argument(
        "--logs_path",
        default="./yolov5_log/",  
    )
    
    parser.add_argument(
        "--out_path",
        default="./yolov5_out",  
    )

    parser.add_argument(
        "--serialize",
        default="./yolov5_serialize/",
    )
    
    args = parser.parse_args()
    return args

class Infer():
    def __init__( self,
        device,
        batch_size,
        model_path,
        logs_path,
        serialize):
        
        print("suinfer session")
        self.logs_path = logs_path
        self.serialize = serialize
        self.batch_size = batch_size
        self.device = device
        self.model_path = model_path
        self.input_shape, self.output_name = self.get_shapeinfo()
        self.model_session = self.build()
        
    
    def get_shapeinfo(self):
        model = onnx.load(self.model_path)
        inputshape = {}
        outputname = []
        
        for input in model.graph.input:
            tmp_shape = []
            for shape in input.type.tensor_type.shape.dim:
                tmp_shape.append(shape.dim_value)
            tmp_shape[0] = self.batch_size
            inputshape[input.name] = tmp_shape
        
        for output in model.graph.output:
            outputname.append(output.name)
        return inputshape, outputname
        
    
    def build(self):
        model_session = pysuInferSession(self.device, self.logs_path + "model.log")
        policy = []
        model_serialize = os.path.join(self.serialize, "model.bin")
        
        if os.path.exists(model_serialize):
            print("build model from serialize, model path:", model_serialize)
            res = model_session.Init(model_serialize)
        else:    
            print("start to build model, model path: ", self.model_path)
            res = model_session.Build(
                self.model_path,
                self.input_shape,
                model_serialize,
                policy=policy,
            )

        if res != suResult.SUI_STATUS_SUCCESS:
            print("Failed to build model")
            return None
        print("end to build model")
        
        return model_session
    
    def preparedata(self, datapath, data_name, input_shape, dtype):
        batch_input = np.ones(input_shape, dtype)
        single_shape = copy.deepcopy(input_shape)
        single_shape[0] = 1
        for idx, i in enumerate(data_name):
            single_path = os.path.join(datapath, i)
            single_data = np.fromfile(single_path,dtype).reshape(single_shape)
            batch_input[idx] = single_data
        return batch_input
        
        
        
    def rundata(self, datapath, out_path, dtype):
        files = os.listdir(datapath)
        batch_input = {}    
        for batch_idx in tqdm(range(0, len(files), self.batch_size)):
            batch_name = files[batch_idx : batch_idx + self.batch_size]
            for input in self.input_shape:
                batch_input[input] = self.preparedata(datapath, batch_name, self.input_shape[input], dtype)

            batch_output = prepareOutput(
                self.model_session, self.output_name
            )
            
            runSession(
                self.model_session,
                batch_input,
                batch_output
            )

            for idx, outname in enumerate(batch_name):
                for i in batch_output:
                    batch_output[i][idx].tofile(os.path.join(out_path, outname + ".bin"))
    

def main():
    args = get_params()

    GetVersion()

    predictor = Infer(
        int(args.device),
        args.batchsize,
        args.model_path,
        args.logs_path,
        args.serialize 
    )
    
    predictor.rundata(args.ori_data, args.out_path, np.float32)

    
if __name__ == "__main__":
    main()

