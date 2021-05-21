
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2
import torch

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
# from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm
# import mmcv
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.structures import Boxes, Instances
# from detectron2.projects.BorderMask import add_bordermask_config

from lod_net import add_lod_config
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
)
from detectron2.engine import DefaultPredictor
from thop import profile
from thop import clever_format

def dataset_register():
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("ETIS_val", {}, "../../datasets/ETIS-LaribPolypDB/annotations/instances_val2017.json",
                            "../../datasets/ETIS-LaribPolypDB/val2017")

    register_coco_instances("CVC_ClinicDB_val", {}, "../../datasets/CVC-ClinicDB/annotations/instances_val2017.json",
                            "../../datasets/CVC-ClinicDB/val2017")

    register_coco_instances("CVC_ColonDB_val", {}, "../../datasets/CVC-ColonDB/annotations/instances_val2017.json",
                            "../../datasets/CVC-ColonDB/val2017")

    register_coco_instances("CVC_300_val", {}, "../../datasets/CVC-300/annotations/instances_val2017.json",
                            "../../datasets/CVC-300/val2017")

    register_coco_instances("Kvasir_val", {}, "../../datasets/Kvasir/annotations/instances_val2017.json",
                            "../../datasets/Kvasir/val2017")

    return ["CVC_ClinicDB_val", "ETIS_val", "Kvasir_val", "CVC_ColonDB_val", "CVC_300_val"]

datasets = dataset_register()
cfg = get_cfg()
add_lod_config(cfg)
# cfg.merge_from_file("./config/LOD_test.yaml")
cfg.merge_from_file("../../configs/polyp/mask_rcnn_R_101_FPN_1x_class1.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "../../polyp/mask_rcnn_R_101_FPN_1x_class1/model_final.pth"
visPath = "../output/debug/"

predictor = DefaultPredictor(cfg)


model = predictor.model
# for k, v in model.named_parameters():
#     print(k, v.size())

data_loader = build_detection_test_loader(cfg, "ETIS_val")

for idx, inputs in tqdm(enumerate(data_loader)):

    if idx>0:
        break
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    # flops, params = profile(model, inputs=(data,))
    flops, params = clever_format([flops, params], "%.7f")
    print(flops, params)



