import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
from argparse import Namespace
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(
    0, "/data4/ersp2022/models/GRiT/third_party/CenterNet2/projects/CenterNet2"
)  # Add path to CenterNet2
from centernet.config import add_centernet_config
from grit.config import add_grit_config
import numpy as np

import scipy.stats as stats
from grit.predictor import VisualizationDemo


# constants
WINDOW_NAME = "GRiT"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default="",
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def run_grit_VIP(images_folder: str):
    "Runs grit to return detected objects and confidence scores"
    output = {}
    args = Namespace(
        config_file="/data4/ersp2022/models/GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml",  # Add path to GriT config
        cpu=False,
        input=[f"{images_folder}"],
        output="visualization",
        confidence_threshold=0.5,
        test_task="DenseCap",
        opts=[
            "MODEL.WEIGHTS",
            "/data4/ersp2022/models/GRiT/models/grit_b_densecap_objectdet.pth",
        ],
    )
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    for path in tqdm.tqdm(os.listdir(args.input[0]), disable=not args.output):
        img = read_image(os.path.join(args.input[0], path), format="BGR")
        predictions = demo.run_on_image(img)
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        scores = stats.zscore(scores.cpu().numpy())
        object_description = predictions.pred_object_descriptions.data
        output[path] = ""
        for d, score in zip(object_description, scores):
            for row in boxes:
                x1 = row[0]
                y1 = row[1]
                x2 = row[2]
                y2 = row[3]
            output[
                path
            ] += f"{d} x1: {x1:.2f} x2: {x2:.2f} y1: {y1:.2f} y2: {y2:.2f} score: {score:.3f}\n"
    return output


def runGRiTKeyFrame_VIP(images_folder: str):
    """Runs Grit and simply return detected objects in one sentence."""
    output = {}
    args = Namespace(
        config_file="/data4/ersp2022/models/GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml",  # Add path to GRiT config
        cpu=False,
        input=[images_folder],
        output="visualization",
        confidence_threshold=0.5,
        test_task="DenseCap",
        opts=[
            "MODEL.WEIGHTS",
            "/data4/ersp2022/models/GRiT/models/grit_b_densecap_objectdet.pth",
        ],
    )
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    for path in tqdm.tqdm(os.listdir(args.input[0]), disable=not args.output):
        img = read_image(os.path.join(args.input[0], path), format="BGR")
        predictions = demo.run_on_image(img)
        scores = predictions.scores if predictions.has("scores") else None
        scores = stats.zscore(scores.cpu().numpy())
        object_description = predictions.pred_object_descriptions.data
        objectListString = ""
        for i in range(len(object_description)):
            objectListString += f"{object_description[i]}"
            if i == (len(object_description) - 1):
                objectListString += "."
            else:
                objectListString += ", "
        output[path] = objectListString
    return output


if __name__ == "__main__":
    print(
        run_grit_VIP("/data4/ersp2022/danny/musharna/keyframes/979838")
    )  # Add path to images folder
