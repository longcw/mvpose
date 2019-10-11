from torch.utils.data import Dataset
import numpy as np
import re
from glob import glob
import pickle
import os
import os.path as osp
from collections import OrderedDict
import cv2


class BaseDataset(Dataset):
    def __init__(self, dataset_dir, range_, with_name=False):
        abs_dataset_dir = osp.abspath(dataset_dir)
        self.image_root = osp.join(abs_dataset_dir, "frames")
        self.with_name = with_name

        cam_dirs = [
            i
            for i in sorted(glob(osp.join(self.image_root, "*/")))
            if re.search(r"\d+/$", i)
        ]
        self.infos = OrderedDict()
        self.camera_ids = []
        for cam_idx, cam_dir in enumerate(cam_dirs):
            # cam_id = int(re.search(r"\d+/$", cam_dir).group().strip("/"))
            self.camera_ids.append(cam_idx)

            self.infos[cam_idx] = OrderedDict()
            img_lists = sorted(glob(osp.join(cam_dir, "*")))
            if range_ is None:
                range_ = range(len(img_lists))

            for i, img_id in enumerate(range_):
                img_path = img_lists[img_id]
                self.infos[cam_idx][i] = img_path

    def __len__(self):
        return len(self.infos[0])

    def __getitem__(self, item):
        imgs = list()
        image_names = []
        for cam_id in self.infos.keys():
            # imgs.append ( cv2.imread ( cam_infos[item] ) )
            image_names.append(
                os.path.relpath(self.infos[cam_id][item], self.image_root)
            )
            imgs.append(cv2.imread(self.infos[cam_id][item]))

        if self.with_name:
            return imgs, image_names
        else:
            return imgs


class PreprocessedDataset(Dataset):
    def __init__(self, dataset_dir, start=0, step=1):
        self.abs_dataset_dir = osp.abspath(dataset_dir)
        self.info_files = sorted(
            glob(osp.join(self.abs_dataset_dir, "*.pickle")),
            key=lambda x: int(osp.basename(x).split(".")[0]),
        )  # To take %d.infodicts.pickle

        sorted(self.info_files)
        if start > 0 or step > 1:
            self.info_files = [
                self.info_files[i] for i in range(start, len(self.info_files), step)
            ]

    def __len__(self):
        return len(self.info_files)

    def __getitem__(self, item):
        with open(self.info_files[item], "rb") as f:
            info_dicts = pickle.load(f)
        dump_dir = self.abs_dataset_dir
        img_id = int(osp.basename(self.info_files[item]).split(".")[0])
        for cam_id, info_dict in info_dicts.items():
            info_dict["image_data"] = np.load(
                osp.join(dump_dir, info_dict["image_path"])
            )

            for pid, person in enumerate(info_dict[0]):
                person.update(np.load(osp.join(dump_dir, person["data_path"])))
                # person['heatmap_data'] = np.load(osp.join(dump_dir, person['heatmap_path']))
                # person['cropped_img'] = np.load(osp.join(dump_dir, person['cropped_path']))
        return info_dicts
