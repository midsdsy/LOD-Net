
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from matplotlib import pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
# from mmdet.apis import inference_detector, init_detector
# import mmcv
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.structures import Boxes, Instances
from border_mask import add_bordermask_config
from tqdm import tqdm
import matplotlib

def dataset_register():
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("ETIS_train", {}, "../../datasets/ETIS-LaribPolypDB/annotations/instances_train2017.json",
                            "../../datasets/ETIS-LaribPolypDB/train2017")
    register_coco_instances("ETIS_val", {}, "../../datasets/ETIS-LaribPolypDB/annotations/instances_val2017.json",
                            "../../datasets/ETIS-LaribPolypDB/val2017")

    # register_coco_instances("CVC_ClinicDB_train", {}, "datasets/CVC-ClinicDB/annotations/instances_train2017.json",
    #                         "datasets/CVC-ClinicDB/train2017")
    register_coco_instances("CVC_ClinicDB_val", {}, "../../datasets/CVC-ClinicDB/annotations/instances_val2017.json",
                            "../../datasets/CVC-ClinicDB/val2017")

    # register_coco_instances("CVC_ColonDB_train", {}, "datasets/CVC-ColonDB/annotations/instances_train2017.json",
    #                         "datasets/CVC-ColonDB/train2017")
    register_coco_instances("CVC_ColonDB_val", {}, "../../datasets/CVC-ColonDB/annotations/instances_val2017.json",
                            "../../datasets/CVC-ColonDB/val2017")

    # register_coco_instances("CVC_300_train", {}, "datasets/CVC-300/annotations/instances_train2017.json",
    #                         "datasets/CVC-300/train2017")
    register_coco_instances("CVC_300_val", {}, "../../datasets/CVC-300/annotations/instances_val2017.json",
                            "../../datasets/CVC-300/val2017")

    # register_coco_instances("Kvasir_train", {}, "datasets/Kvasir/annotations/instances_train2017.json",
    #                         "datasets/Kvasir/train2017")
    register_coco_instances("Kvasir_val", {}, "../../datasets/Kvasir/annotations/instances_val2017.json",
                            "../../datasets/Kvasir/val2017")

import detectron2.data.transforms as T
from border_mask import MotionBlur

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.BLUR.ENABLED:
        augs.append(MotionBlur(prob=cfg.INPUT.BLUR.Prob,kernel=cfg.INPUT.BLUR.KERNEL_SIZE))
    augs.append(T.RandomFlip())
    return augs

from detectron2.data import DatasetMapper
def build_train_loader(cfg):

    if cfg.INPUT.AUG:
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
    else:
        mapper = None
    return build_detection_train_loader(cfg, mapper=mapper)

cfg = get_cfg()
dataset_register()
add_bordermask_config(cfg)
cfg.merge_from_file("config/polyp/bordermask_LOG_R_101_FPN.yaml")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = "../../polyp/bordermask_LOG_R_101_FPN/model_final.pth"
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.INPUT.AUG = True
cfg.INPUT.BLUR.ENABLED = True
cfg.INPUT.BLUR.Prob = 0.7
cfg.INPUT.BLUR.KERNEL_SIZE = 3


predictor = DefaultPredictor(cfg)
model = predictor.model

meta = MetadataCatalog.get("ETIS_train")

visPath = "../../polyp_lodaer/bordermask_LOG_R_101_FPN/train_lodaer"
import os

if not os.path.isdir(visPath):
    os.makedirs(visPath)

data_loader = build_train_loader(cfg)
# data_loader = build_detection_train_loader(cfg)
for idx, inputs in tqdm(enumerate(data_loader)):
    if idx>0:
        break
    file_name = inputs[0]['file_name']
    # origin_img = cv2.imread(file_name)
    base_name = file_name.split("/")[-1]
    images = model.preprocess_image(inputs).tensor  #[B,3 H,W]
    # import pdb
    # pdb.set_trace()
    images =images.permute(0,2,3,1)

    processed_image= images[0,:,:,:].cpu().numpy()
    # import pdb
    # pdb.set_trace()
    cv2.imwrite(os.path.join(visPath, "aug_" + base_name), processed_image)


    plt.imshow(processed_image)
    plt.show()



