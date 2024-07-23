# Copyright (c) OpenMMLab. All rights reserved.
import torch
from queue import Queue
from threading import Thread
from typing import Tuple
from torch import distributed as dist

def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

class CudaDataLoader:
    def __init__(self,loader,device,queue_size=2):
        self.device = device
        self.queue_size=queue_size
        self.loader=loader

        self.load_stream = dist.get_group_stream()
        self.queue = Queue(maxsize=self.queue_size)
        self.idx=0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()
        self.num_workers = loader.num_workers

    def load_loop(self):
        while True:
            for i, data in enumerate(self.loader):
                # print("data length", len(data))
                imgs, targets, files, shape = data
                # print("imgs", imgs)
                # print("targets", targets)
                self.queue.put(self.load_instance(imgs, targets, files, shape))

    def load_instance(self, imgs, targets, files, shape):
        with torch.supa.stream(self.load_stream):
            images = imgs.to(self.device,non_blocking=True)
            targets = targets.to(self.device,non_blocking=True)

        return [images,targets, files, shape]

    def __iter__(self):
        self.idx =0
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx=0
            self.queue.join()
            self.worker.join()
            raise StopIteration

        elif self.idx >= len(self.loader):
            self.idx=0
            raise StopIteration

        else:
            out= self.queue.get()
            self.queue.task_done()
            self.idx+=1

        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
