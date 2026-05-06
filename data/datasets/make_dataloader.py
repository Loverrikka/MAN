import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import pdb

from .bases import ImageDataset,ImageDataset_MMT,ImageDataset_spcl, ImageDataset_w_str,ImageDatasetCameraAware,ImageDatasetPreprocessor,ImageDatasetsecret
from .sampler import RandomIdentitySampler,PartRandomMultipleGallerySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .msvr310 import MSVR310
from .RGBNT201 import RGBNT201
from .RGBNT100 import RGBNT100
from .MSVwild863 import MSVwild863
from .market_to_RGBNT201 import market_to_RGBNT201
from .MSVR310_neo import MSVR310_neo
from .MSVwild863_neo import MSVwild863_neo
from .MSVR310_neo_100 import MSVR310_neo_100
from .alldayRNTG import AllDayRNTG

from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from libs.utils.data.sampler import ClassUniformlySampler, RandomMultipleGallerySampler, ClusterProxyBalancedSampler
from libs.utils.data.preprocessor import Preprocessor, CameraAwarePreprocessor
from RTMem.utils.data.sampler import RandomMultipleGallerySampler_RTMem

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'RGBNT201': RGBNT201,
    'RGBNT100': RGBNT100,
    'MSVR310': MSVR310,
    'MSVwild863' :MSVwild863,
    'market_to_RGBNT201' :market_to_RGBNT201,
    'MSVR310_neo' : MSVR310_neo,
    'MSVwild863_neo' : MSVwild863_neo,
    'MSVR310_neo_100' : MSVR310_neo_100,
    'alldayRNTG' : AllDayRNTG ,
    
}
""" Random Erasing (Cutout)

Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng

Hacked together by / Copyright 2019, Ross Wightman
"""
import random
import math

import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5,
            min_area=0.02,
            max_area=1 / 3,
            min_aspect=0.3,
            max_aspect=None,
            mode='const',
            min_count=1,
            max_count=None,
            num_splits=0,
            device='cuda',
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if self.mode == 'rand':
            self.rand_color = True  # per block random normal
        elif self.mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not self.mode or self.mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, viewids,_

def train_collate_fn_secret(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, fname, pids, index = zip(*batch)
    # pids = torch.tensor(pids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, fname, pids, index

def train_collate_fn_TMGF(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, img_path, pids, camids, img_index, accum_label = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, img_path, pids, camids,img_index, accum_label

def train_collate_fn_RTMem(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, _, pids, camids, viewids, indices = zip(*batch) #img, img_path, pid, camid, trackid, indices   
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, viewids, indices

def train_collate_fn_MMT(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, imgs1, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    RGB_list1 = []
    NI_list1 = []
    TI_list1 = []
    
    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    
    for img in imgs1:
        RGB_list1.append(img[0])
        NI_list1.append(img[1])
        TI_list1.append(img[2])

    RGB = torch.stack(RGB_list1, dim=0)
    NI = torch.stack(NI_list1, dim=0)
    TI = torch.stack(TI_list1, dim=0)
    imgs1 = {'RGB': RGB, "NI": NI, "TI": TI}
    
    return imgs, imgs1, pids, camids, viewids,_

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    RGB_list = []
    NI_list = []
    TI_list = []

    for img in imgs:
        RGB_list.append(img[0])
        NI_list.append(img[1])
        TI_list.append(img[2])

    RGB = torch.stack(RGB_list, dim=0)
    NI = torch.stack(NI_list, dim=0)
    TI = torch.stack(TI_list, dim=0)
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    target_dataset = __factory[cfg.TARGET_DATASETS.NAMES](root=cfg.TARGET_DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_neo = ImageDataset_w_str(dataset.train, val_transforms, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    target_train_set = ImageDataset(target_dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    if cfg.dataloader_iter :
        # 源域强增强
        train_loader = IterLoader(DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
            ),length=cfg.iters)
    
    else:
        train_loader = DataLoader( 
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )

    source_train_loader = DataLoader(
            train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
        
    target_train_loader = DataLoader(
            target_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
    
    
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    target_val_set = ImageDataset(target_dataset.query + target_dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    target_val_loader = DataLoader(
        target_val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return dataset, target_dataset, train_loader, source_train_loader, target_train_loader, val_loader, target_val_loader, len(dataset.query), len(target_dataset.query), num_classes, cam_num, view_num

def make_cluster_dataloader(cfg,dataset_train,str=True, sampler=True):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    train_transforms_863 = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
    
    if str:
        # if cfg.TARGET_DATASETS.NAMES == 'MSVwild863_neo': 
        #     train_set = ImageDataset(dataset_train, train_transforms_863)
        # else:
        #     train_set = ImageDataset(dataset_train, train_transforms)  
                      
        train_set = ImageDataset(dataset_train, train_transforms)            

    else:
        train_set = ImageDataset(dataset_train, val_transforms)

    if sampler:
        if cfg.dataloader_iter :
            train_loader = IterLoader(DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
            ),length=cfg.iters)
            
        else:
            train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
            )
    else:
            train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
            )
        
    return train_loader

def make_cluster_dataloader_secret(cfg,dataset_train,str=True):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if str:
        train_set = ImageDatasetsecret(dataset_train, train_transforms)

    else:
        train_set = ImageDatasetsecret(dataset_train, val_transforms)

    if cfg.dataloader_iter :
        train_loader = IterLoader(DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=PartRandomMultipleGallerySampler(dataset_train, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn_secret, pin_memory=True, drop_last=True
        ),length=cfg.iters)
        
    else:
        train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=PartRandomMultipleGallerySampler(dataset_train, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn_secret, pin_memory=True, drop_last=True
        )
        
    return train_loader

def make_dataloader_DHCCN(cfg):
    train_transforms = T.Compose([
        T.Resize([384,128], interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop([384,128]),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    
    val_transforms = T.Compose([
        T.Resize([384,128]),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    target_dataset = __factory[cfg.TARGET_DATASETS.NAMES](root=cfg.TARGET_DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_neo = ImageDataset_w_str(dataset.train, val_transforms, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    target_train_set = ImageDataset(target_dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    if cfg.dataloader_iter :
        # 源域强增强
        train_loader = IterLoader(DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
            ),length=cfg.iters)
    
    else:
        train_loader = DataLoader( 
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
    
    # 源域强弱增强
    # train_loader = IterLoader(DataLoader(
    #         train_set_neo, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn_MMT, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)
    
    # # 源域弱增强
    # train_loader = IterLoader(DataLoader(
    #         train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)

    source_train_loader = DataLoader(
            train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
        
    target_train_loader = DataLoader(
            target_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
    
    
    # source_train_loader = IterLoader(DataLoader(
    #         train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)
        
    # target_train_loader = IterLoader(DataLoader(
    #         target_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)
    
    
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    target_val_set = ImageDataset(target_dataset.query + target_dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    target_val_loader = DataLoader(
        target_val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return dataset, target_dataset, train_loader, source_train_loader, target_train_loader, val_loader, target_val_loader, len(dataset.query), len(target_dataset.query), num_classes, cam_num, view_num

def make_dataloader_LRIMV(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    target_dataset = __factory[cfg.TARGET_DATASETS.NAMES](root=cfg.TARGET_DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    target_train_set = ImageDataset(target_dataset.train, val_transforms)
    t_trains = []
    for i in range(cfg.LRIMV.CAMERA_NUM):
        t_trains.append(ImageDataset(target_dataset.c_trains[i], val_transforms))
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    if cfg.dataloader_iter :
        # 源域强增强
        train_loader = IterLoader(DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
            ),length=cfg.iters)
    
    else:
        train_loader = DataLoader( 
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
    
    # 源域强弱增强
    # train_loader = IterLoader(DataLoader(
    #         train_set_neo, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn_MMT, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)
    
    # # 源域弱增强
    # train_loader = IterLoader(DataLoader(
    #         train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)

    source_train_loader = DataLoader(
            train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
        
    target_train_loader = DataLoader(
            target_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False,
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )
    t_loader = []
    for i in range(cfg.LRIMV.CAMERA_NUM):
        t_loader.append(DataLoader(
            t_trains[i], batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=num_workers, collate_fn=train_collate_fn
        ))    
    pdb.set_trace()
    # source_train_loader = IterLoader(DataLoader(
    #         train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)
        
    # target_train_loader = IterLoader(DataLoader(
    #         target_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #         sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
    #         num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
    #     ),length=cfg.iters)
    
    
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    target_val_set = ImageDataset(target_dataset.query + target_dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    target_val_loader = DataLoader(
        target_val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return dataset, target_dataset, train_loader, source_train_loader, target_train_loader, t_loader, val_loader, target_val_loader, len(dataset.query), len(target_dataset.query), num_classes, cam_num, view_num

def make_cluster_dataloader_LRIMV(cfg,dataset_train,camera_intra_train_loader,t_loader):
    train_transforms = T.Compose([
        T.Resize([256,128], interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop([256,128]),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    val_transforms = T.Compose([
        T.Resize([256,128]),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
                  
    camera_intra_train_set = ImageDataset(dataset_train, train_transforms)          
      
    camera_intra_train_loader.append(DataLoader(
        camera_intra_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, 4),
        num_workers=num_workers, collate_fn=train_collate_fn
    ))

    t_datat_train_set = ImageDataset(dataset_train, val_transforms)
    t_loader.append(DataLoader(
        t_datat_train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        num_workers=num_workers, collate_fn=train_collate_fn
    ))

        
    return camera_intra_train_loader,t_loader


def make_cluster_dataloader_DHCCN(cfg,dataset_train,str=True):
    train_transforms = T.Compose([
        T.Resize([384,128], interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop([384,128]),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    val_transforms = T.Compose([
        T.Resize([384,128]),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
    
    if str:                    
        train_set = ImageDataset(dataset_train, train_transforms)            

    else:
        train_set = ImageDataset(dataset_train, val_transforms)
            
    if cfg.dataloader_iter :
        train_loader = IterLoader(DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
        ),length=cfg.iters)
        
    else:
        train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=True
        )
        
    return train_loader

def make_cluster_loader_TMGF(cfg, trainset=None):
    # Preprocessing
    normalizer = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                             std=cfg.INPUT.PIXEL_STD)
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    train_set = sorted(trainset)

    # Choose sampler type
    # class_position [1: cluster_label, 4: proxy_label]
    if cfg.TMGF.SAMPLER.TYPE == 'proxy_balance':
        sampler = ClassUniformlySampler(train_set, class_position=4, k=cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.TMGF.SAMPLER.TYPE == 'cluster_balance':
        sampler = ClassUniformlySampler(train_set, class_position=1, k=cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.TMGF.SAMPLER.TYPE == 'cam_cluster_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=1, num_instances=cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.TMGF.SAMPLER.TYPE == 'cam_proxy_balance':
        sampler = RandomMultipleGallerySampler(train_set, class_position=4, num_instances=cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.TMGF.SAMPLER.TYPE == 'cluster_proxy_balance':
        sampler = ClusterProxyBalancedSampler(train_set, k=cfg.DATALOADER.NUM_INSTANCE)
    else:
        raise ValueError('Invalid sampler type name!')

    # Create dataloader
    train_loader = IterLoader(
                DataLoader(ImageDatasetCameraAware(train_set, transform=train_transforms),
                            batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn_TMGF, sampler=sampler,
                            shuffle=False, pin_memory=True, drop_last=True), length=cfg.iters)
    return train_loader

def make_cluster_loader_RTMem(cfg, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    train_set = sorted(trainset)
    sampler = RandomMultipleGallerySampler_RTMem(train_set, num_instances=cfg.DATALOADER.NUM_INSTANCE)
    train_loader = IterLoader(
        DataLoader(ImageDatasetPreprocessor(train_set, transform=train_transformer),
                   batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn_RTMem, sampler=sampler,
                   shuffle=False, pin_memory=True, drop_last=True), length=cfg.iters)

    return train_loader

def make_cluster_dataloader_get_image(cfg,dataset_train):
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageDataset(dataset_train, val_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=None,num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True, drop_last=False
        )
        
    return train_loader


def make_cluster_dataloader_MMT(cfg,dataset_train):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = sorted(dataset_train)
    dataset_train = ImageDataset_MMT(train_set, train_transforms)

    if cfg.dataloader_iter :
        train_loader = IterLoader(DataLoader(
        dataset_train, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(train_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn_MMT, pin_memory=True, drop_last=True
        ),length=cfg.iters)
        
    else:
        train_loader = DataLoader(
        dataset_train, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(train_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn_MMT, pin_memory=True, drop_last=True
        )
    return train_loader

class ProbUncertain():
    def __init__(self, alpha=20, epsilon=0.99):
        self.alpha = alpha
        self.epsilon = epsilon
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.kl_loss = torch.nn.KLDivLoss(reduction='none')
    
    def cal_uncertainty(self, features, pseudo_labels, classifier):
        features, classifier = torch.from_numpy(features), torch.from_numpy(classifier)
        pred_probs =  self.logsoftmax(self.alpha * torch.matmul(features, classifier.t()))

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
        ideal_probs = torch.zeros(pred_probs.shape) + (1-self.epsilon) / (pred_probs.shape[1]-1)
        ideal_probs.scatter_(1, pseudo_labels.unsqueeze(-1), value=self.epsilon)

        uncertainties = self.kl_loss(pred_probs, ideal_probs).sum(1).numpy()
        return uncertainties


prob_uncertainty = ProbUncertain()
def make_cluster_dataloader_P2LR(cfg, dataset, centers, target_label, cf, pt):

    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    
    uncertainties = prob_uncertainty.cal_uncertainty(cf, target_label, centers)
    N = len(uncertainties) 
    beta = np.sort(uncertainties)[int(pt * N) - 1]
    Vindicator = [False for _ in range(N)]
    for i in range(N):
        if uncertainties[i] <= beta:
            Vindicator[i] = True
    Vindicator = np.array(Vindicator)
    select_samples_inds = np.where(Vindicator == True)[0]
    select_samples_labels = target_label[select_samples_inds]
    train_set = [dataset.train[ind] for ind in select_samples_inds]

    # change pseudo labels
    for i in range(len(train_set)):
        train_set[i] = list(train_set[i])
        train_set[i][1] = int(select_samples_labels[i])
        train_set[i] = tuple(train_set[i])

    print('select {}/{} samples'.format(len(train_set), N))

    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = sorted(train_set)
    dataset_train = ImageDataset_MMT(train_set, train_transforms)
    
    if cfg.dataloader_iter :
        train_loader = IterLoader(DataLoader(
        dataset_train, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(train_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn_MMT, pin_memory=True, drop_last=True
        ),length=cfg.iters)
        
    else:
        train_loader = DataLoader(
        dataset_train, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(train_set, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn_MMT, pin_memory=True, drop_last=True
        )

    return train_loader, select_samples_inds, select_samples_labels

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)