import os
import sys
import os.path as osp
import pickle
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if __name__ == "__main__":
    if project_root not in sys.path:
        sys.path.append(project_root)
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)

from src.models.model_config import model_cfg
import time
import scipy.io as scio
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from prettytable import PrettyTable
import csv
from copy import deepcopy
import torch
from glob import glob
import cv2
from torch.utils.data import DataLoader
from src.m_utils.base_dataset import BaseDataset, PreprocessedDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.transformation import coco2shelf3D
from src.m_utils.numeric import vectorize_distance
from src.m_utils.mem_dataset import MemDataset


def is_right(
    model_start_point, model_end_point, gt_strat_point, gt_end_point, alpha=0.5
):
    bone_lenth = np.linalg.norm(gt_end_point - gt_strat_point)
    start_difference = np.linalg.norm(gt_strat_point - model_start_point)
    end_difference = np.linalg.norm(gt_end_point - model_end_point)
    return ((start_difference + end_difference) / 2) <= alpha * bone_lenth


def numpify(info_dicts):
    for info_dict in info_dicts.values():
        info_dict["image_data"] = info_dict["image_data"].squeeze().numpy()
        for person in info_dict[0]:
            person["heatmap_data"] = person["heatmap_data"].squeeze().numpy()
            person["cropped_img"] = person["cropped_img"].squeeze().numpy()
    return info_dicts


def coco2shelf2D(coco_pose):
    """
    transform coco order(our method output) 3d pose to shelf dataset order with interpolation
    :param coco_pose: np.array with shape 17x2
    :return: 3D pose in shelf order with shape 14x2
    """
    shelf_pose = np.zeros((14, 2))
    coco2shelf = np.array([16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9])
    shelf_pose[0:12] += coco_pose[coco2shelf]

    # Use middle of shoulder to init
    shelf_pose[12] = (shelf_pose[8] + shelf_pose[9]) / 2
    # shelf_pose[13] = coco_pose[0]  # use nose to init
    shelf_pose[13] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array(
        [0, 1.5]
    )
    shelf_pose[12] = shelf_pose[12] + (coco_pose[0] - shelf_pose[12]) * np.array(
        [0, 0.6]
    )

    return shelf_pose


def evaluate(model, actor3D, range_, loader, is_info_dicts=False, dump_dir=None):
    check_result = np.zeros((len(actor3D[0]), len(actor3D), 10), dtype=np.int32)
    accuracy_cnt = 0
    error_cnt = 0
    results_3d = {}  # fid -> p3ds
    results_2d = {}  # camera -> {fid -> p2ds}
    for idx, imgs in enumerate(tqdm(loader)):
        img_id = range_[idx]
        try:
            if is_info_dicts:
                info_dicts = numpify(imgs)
                model.dataset = MemDataset(
                    info_dict=info_dicts,
                    camera_parameter=camera_parameter,
                    template_name="Unified",
                )
                poses3d, pose_mat, matched_list, imgid2cam = model._estimate3d(
                    0, show=False, rtn_2d=True
                )
            else:
                this_imgs = list()
                for img_batch in imgs:
                    this_imgs.append(img_batch.squeeze().numpy())
                poses3d, pose_mat, matched_list, imgid2cam = model.predict(
                    imgs=this_imgs,
                    camera_parameter=camera_parameter,
                    template_name="Unified",
                    show=False,
                    rtn_2d=True,
                )
        except Exception as e:
            logger.critical(e)
            poses3d = False

        # save p3ds
        if not poses3d:
            shelf_poses = []
        else:
            shelf_poses = np.stack([coco2shelf3D(i) for i in deepcopy(poses3d)])
        results_3d[img_id] = []
        for i, p3ds in enumerate(shelf_poses):
            p3ds = p3ds.astype(float) * 100.0
            results_3d[img_id].append(
                {"frame": img_id, "id": i, "points_3d": p3ds.tolist()}
            )

        # save 2d
        for i, matched in enumerate(matched_list):
            for pose_i in matched:
                camera_id = int(imgid2cam[pose_i])
                results_2d.setdefault(camera_id, {})
                results_2d[camera_id].setdefault(img_id, [])
                shelf_pose = coco2shelf2D(pose_mat[pose_i].astype(float))
                results_2d[camera_id][img_id].append(
                    {
                        "camera": camera_id,
                        "frame": img_id,
                        "id": i,
                        "points_2d": shelf_pose.tolist(),
                    }
                )

        for pid in range(len(actor3D)):
            if actor3D[pid][img_id][0].shape == (1, 0) or actor3D[pid][img_id][
                0
            ].shape == (0, 0):

                continue

            if not poses3d:
                check_result[img_id, pid, :] = -1
                logger.error(f"Cannot get any pose in img:{img_id}")
                continue
            model_poses = np.stack([coco2shelf3D(i) for i in deepcopy(poses3d)])
            gt_pose = actor3D[pid][img_id][0]
            dist = vectorize_distance(np.expand_dims(gt_pose, 0), model_poses)
            model_pose = model_poses[np.argmin(dist[0])]

            bones = [
                [0, 1],
                [1, 2],
                [3, 4],
                [4, 5],
                [6, 7],
                [7, 8],
                [9, 10],
                [10, 11],
                [12, 13],
            ]
            for i, bone in enumerate(bones):
                start_point, end_point = bone
                if is_right(
                    model_pose[start_point],
                    model_pose[end_point],
                    gt_pose[start_point],
                    gt_pose[end_point],
                ):
                    check_result[img_id, pid, i] = 1
                    accuracy_cnt += 1
                else:
                    check_result[img_id, pid, i] = -1
                    error_cnt += 1
            gt_hip = (gt_pose[2] + gt_pose[3]) / 2
            model_hip = (model_pose[2] + model_pose[3]) / 2
            if is_right(model_hip, model_pose[12], gt_hip, gt_pose[12]):
                check_result[img_id, pid, -1] = 1
                accuracy_cnt += 1
            else:
                check_result[img_id, pid, -1] = -1
                error_cnt += 1
    bone_group = OrderedDict(
        [
            ("Head", np.array([8])),
            ("Torso", np.array([9])),
            ("Upper arms", np.array([5, 6])),
            ("Lower arms", np.array([4, 7])),
            ("Upper legs", np.array([1, 2])),
            ("Lower legs", np.array([0, 3])),
        ]
    )

    total_avg = np.sum(check_result > 0) / np.sum(np.abs(check_result))
    person_wise_avg = np.sum(check_result > 0, axis=(0, 2)) / np.sum(
        np.abs(check_result), axis=(0, 2)
    )

    bone_wise_result = OrderedDict()
    bone_person_wise_result = OrderedDict()
    for k, v in bone_group.items():
        bone_wise_result[k] = np.sum(check_result[:, :, v] > 0) / np.sum(
            np.abs(check_result[:, :, v])
        )
        bone_person_wise_result[k] = np.sum(
            check_result[:, :, v] > 0, axis=(0, 2)
        ) / np.sum(np.abs(check_result[:, :, v]), axis=(0, 2))

    tb = PrettyTable()
    tb.field_names = (
        ["Bone Group"]
        + [f"Actor {i}" for i in range(bone_person_wise_result["Head"].shape[0])]
        + ["Average"]
    )
    list_tb = [tb.field_names]
    for k, v in bone_person_wise_result.items():

        this_row = (
            [k]
            + [np.char.mod("%.4f", i) for i in v]
            + [np.char.mod("%.4f", np.sum(v) / len(v))]
        )
        list_tb.append(
            [float(i) if isinstance(i, type(np.array([]))) else i for i in this_row]
        )
        tb.add_row(this_row)
    this_row = (
        ["Total"]
        + [np.char.mod("%.4f", i) for i in person_wise_avg]
        + [np.char.mod("%.4f", np.sum(person_wise_avg) / len(person_wise_avg))]
    )
    tb.add_row(this_row)
    list_tb.append(
        [float(i) if isinstance(i, type(np.array([]))) else i for i in this_row]
    )
    if dump_dir:
        np.save(
            osp.join(
                dump_dir,
                time.strftime(
                    str(model_cfg.testing_on) + "_%Y_%m_%d_%H_%M",
                    time.localtime(time.time()),
                ),
            ),
            check_result,
        )
        with open(
            osp.join(
                dump_dir,
                time.strftime(
                    str(model_cfg.testing_on) + "_%Y_%m_%d_%H_%M.csv",
                    time.localtime(time.time()),
                ),
            ),
            "w",
        ) as f:
            writer = csv.writer(f)
            writer.writerows(list_tb)
            writer.writerow([model_cfg])
    print(tb)
    print(model_cfg)

    results = {"res_3d": results_3d, "res_2d": results_2d}
    if dump_dir is not None:
        output_file = os.path.join(dump_dir, "results_{}.json".format(dataset_name))
        logging.info("Save results to {}".format(output_file))

        with open(output_file, "w") as f:
            json.dump(results, f)

    return check_result, list_tb


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", nargs="+", dest="datasets", required=True, choices=["Shelf", "Campus"]
    )
    parser.add_argument("-dumped", nargs="+", dest="dumped_dir", default=None)
    parser.add_argument("-workers", type=int, default=4, dest="workers")
    args = parser.parse_args()

    test_model = MultiEstimator(cfg=model_cfg)

    for dataset_idx, dataset_name in enumerate(args.datasets):
        model_cfg.testing_on = dataset_name

        if dataset_name == "Shelf":
            dataset_path = model_cfg.shelf_path
            test_range = range(300, 600)
            gt_path = dataset_path

        elif dataset_name == "Campus":
            dataset_path = model_cfg.campus_path
            test_range = list(range(350, 471)) + list(range(650, 751))
            gt_path = dataset_path

        else:
            dataset_path = model_cfg.panoptic_ultimatum_path
            test_range = range(4500, 4900)
            gt_path = osp.join(dataset_path, "..")

        with open(osp.join(dataset_path, "camera_parameter.pickle"), "rb") as f:
            camera_parameter = pickle.load(f)
        if args.dumped_dir:
            test_dataset = PreprocessedDataset(args.dumped_dir[dataset_idx])
            logger.info(
                f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation"
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                pin_memory=False,
                num_workers=args.workers,
                shuffle=False,
            )
        else:
            test_dataset = BaseDataset(dataset_path, test_range)
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                pin_memory=False,
                num_workers=args.workers,
                shuffle=False,
            )

        actorsGT = scio.loadmat(osp.join(gt_path, "actorsGT.mat"))
        test_actor3D = actorsGT["actor3D"][0]
        if dataset_name == "Panoptic":
            test_actor3D /= 100  # mm->m
        evaluate(
            test_model,
            test_actor3D,
            test_range,
            test_loader,
            is_info_dicts=bool(args.dumped_dir),
            dump_dir=osp.join(project_root, "result"),
        )

