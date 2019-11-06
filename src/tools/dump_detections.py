import os
import sys
import os.path as osp
import json
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)
import numpy as np
from src.models.model_config import model_cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.m_utils.base_dataset import BaseDataset
from src.models.estimate3d import MultiEstimator


def dump_mem(model, loader, dump_dir):
    camera_ids = loader.dataset.camera_ids
    n_frames = len(loader.dataset)
    data = {
        "n_targets": -1,
        "n_frames": n_frames,
        "n_cameras": len(camera_ids),
        "gts_3d": [[] for _ in range(n_frames)],
        "gts_2d": {
            camera_id: [[] for _ in range(n_frames)] for camera_id in camera_ids
        },  # to fit the format of ground truth
        "frame_names": {
            camera_id: [None] * n_frames for camera_id in camera_ids
        },  # to fit the format of ground truth
    }

    for fid, (imgs, image_names) in enumerate(tqdm(loader)):
        # poses3d = model.estimate3d ( img_id=img_id, show=False )
        # inference
        this_imgs = [img_batch.squeeze().numpy() for img_batch in imgs]
        info_dicts = model._infer_single2d(imgs=this_imgs)

        for image_name, (camera_id, info_dict) in zip(image_names, info_dicts.items()):
            # json_name = os.path.splitext(image_name[0])[0] + ".json"
            # path2save = osp.join(dump_dir, json_name)
            # os.makedirs(os.path.dirname(path2save), exist_ok=True)

            # generate dummy detections
            detections = info_dict[0]  # person_id -> {pose2d, bbox}
            for det in detections:
                pose2d = np.asarray(det["pose2d"], dtype=float).reshape(-1, 3)
                person = {
                    "id": -1,
                    "camera": camera_id,
                    "frame": fid,
                    "points_2d": pose2d[:, 0:2].tolist(),
                    "scores": det["scores"],
                }
                data["gts_2d"][camera_id][fid].append(person)
            data["frame_names"][camera_id][fid] = image_name

    with open(osp.join(dump_dir, "mvpose_detections2.json"), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Usage: python preprocess.py -d Shelf Campus Panoptic [-dump_dir ./datasets]"
    )
    parser.add_argument("-d", nargs="+", dest="datasets")
    parser.add_argument("-workers", type=int, default=4, dest="workers")
    args = parser.parse_args()

    test_model = MultiEstimator(cfg=model_cfg)
    # for template_mat in ['h36m', 'Shelf', 'Campus']:
    # for dataset_name in ['Panoptic']:
    for dataset_name in args.datasets:
        # for metric in ['geometry mean', 'Geometry only', 'ReID only']:
        model_cfg.testing_on = dataset_name
        # from backend.CamStyle.reid.common_datasets import load_template

        # template = load_template ( template_mat )
        if dataset_name == "Shelf":
            dataset_path = model_cfg.shelf_path
        elif dataset_name == "Campus":
            dataset_path = model_cfg.campus_path
        elif dataset_name == "28Cam":
            dataset_path = "/data/3DPose/2388walsh_c4_28cams/record4"
        elif dataset_name == "Panoptic":
            dataset_path = model_cfg.panoptic_ultimatum_path
        elif dataset_name == "ultimatum1":
            dataset_path = model_cfg.ultimatum1_path
        elif dataset_name == "HD_ultimatum1":
            dataset_path = model_cfg.HD_ultimatum1_path
        else:
            logger.error(f"Unknown dataset name: {dataset_name}")
            exit(-1)
        # print ( f'Using template on {template_mat}' )
        test_dataset = BaseDataset(dataset_path, range_=None, with_name=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            pin_memory=False,
            num_workers=args.workers,
            shuffle=False,
        )
        # test_dataset.template = template
        # test_model.dataset = test_dataset
        this_dump_dir = dataset_path
        os.makedirs(this_dump_dir, exist_ok=True)
        dump_mem(test_model, test_loader, this_dump_dir)
