from __future__ import division, print_function, absolute_import
from os import pardir
import re
import glob
import os.path as osp
import os

from numpy.lib.twodim_base import tri
from .bases import BaseImageDataset

import warnings



class MSVwild863_neo(BaseImageDataset):
    dataset_dir = 'MSVwild863_neo'

    def __init__(self, root='', verbose=True, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated.'
            )


        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'test')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self._check_before_run()

        self.train = self.process_dir(self.train_dir, relabel=True)
        self.c_trains = self.camera_datasets(self.train_dir, relabel=True)
        self.query = self.process_dir(self.query_dir, relabel=False)
        self.gallery = self.process_dir(self.gallery_dir, relabel=False)
        
        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

        super(MSVwild863_neo, self).__init__()
        
        
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, relabel=False):
        vids = os.listdir(dir_path)
        labels = [int(vid) for vid in vids]
        if relabel:
            label_map = dict()
            for i, lab in enumerate(labels):
                label_map[lab] = i
        data = []
        cam_set = set()
        for vid in vids:
            id_vimgs = os.listdir(os.path.join(dir_path, vid, "vis"))
            id_nimgs = os.listdir(os.path.join(dir_path, vid, "ni"))
            id_timgs = os.listdir(os.path.join(dir_path, vid, "th"))
            for i, img in enumerate(id_vimgs):
                vpath = os.path.join(dir_path, vid, "vis", id_vimgs[i])
                npath = os.path.join(dir_path, vid, "ni", id_nimgs[i])
                tpath = os.path.join(dir_path, vid, "th", id_timgs[i])
                label = label_map[int(vid)] if relabel else int(vid)
                
                # sceneid = re.search('s+\d\d\d',img).group(0)[1:]
                # sceneid = int(sceneid)  # scene id
                sceneid = -1

                night = re.search('n+\d',img).group(0)[1]
                cam = re.search('v+\d',img).group(0)[1]
                cam = int(cam)
                night = int(night)
                cam_set.add(cam)
                data.append(((vpath, npath, tpath), label, cam,sceneid))
        return data

    def camera_datasets(self, dir_path, relabel=False):
        vids = os.listdir(dir_path)
        labels = [int(vid) for vid in vids]
        if relabel:
            label_map = {lab: i for i, lab in enumerate(labels)}
        
        # 初始化6个子数据集的列表
        dataset_c = [[] for _ in range(8)]

        cam_set = set()
        for vid in vids:
            id_vimgs = os.listdir(os.path.join(dir_path, vid, "vis"))
            id_nimgs = os.listdir(os.path.join(dir_path, vid, "ni"))
            id_timgs = os.listdir(os.path.join(dir_path, vid, "th"))
            for i, img in enumerate(id_vimgs):
                vpath = os.path.join(dir_path, vid, "vis", id_vimgs[i])
                npath = os.path.join(dir_path, vid, "ni", id_nimgs[i])
                tpath = os.path.join(dir_path, vid, "th", id_timgs[i])
                label = label_map[int(vid)] if relabel else int(vid)
                
                # sceneid = re.search('s+\d\d\d', img).group(0)[1:]
                # sceneid = int(sceneid)  # scene id
                sceneid = -1

                night = re.search('n+\d', img).group(0)[1]
                cam = re.search('v+\d', img).group(0)[1]
                cam = int(cam)
                night = int(night)
                cam_set.add(cam)
                
                # 根据相机ID（cam - 1）加入对应的子数据集
                dataset_c[cam].append(((vpath, npath, tpath), label, cam, sceneid))
        
        return dataset_c

