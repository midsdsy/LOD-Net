
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
from tools.train_net import Trainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from border_mask import add_bordermask_config


# generate mask images
def dataset_register():
    from detectron2.data.datasets import register_coco_instances

    # register_coco_instances("endocv2021_train", {}, "datasets/EndoCV2021/annotations/instances_train2017.json",
    #                         "datasets/EndoCV2021/train2017")
    # register_coco_instances("endocv2021_val", {}, "datasets/EndoCV2021/annotations/instances_val2017.json",
    #                         "datasets/EndoCV2021/val2017")

    # register_coco_instances("ETIS_train", {}, "datasets/polyp/annotations/instances_train2017.json",
    #                         "datasets/polyp/train2017")
    # register_coco_instances("ETIS_val", {}, "datasets/polyp/annotations/instances_val2017.json",
    #                         "datasets/polyp/val2017")

    # register_coco_instances("ETIS_train", {}, "datasets/ETIS-LaribPolypDB/annotations/instances_train2017.json",
    #                         "datasets/ETIS-LaribPolypDB/train2017")
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

cfg = get_cfg()
add_bordermask_config(cfg)
dataset_register()
cfg.merge_from_file("config/for_sota/bordermask_LOG_R_101_FPN_aug_class1_mstrain.yaml")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # set threshold for this model
cfg.MODEL.WEIGHTS = "../../polyp/final/bordermask_LOG_R_101_FPN_aug_class1_mstrain/model_final.pth"

predictor = DefaultPredictor(cfg)

model = predictor.model


# meta = MetadataCatalog.get("ETIS_val")
print("\n###################### ETIS_val #################\n")
data_loader = build_detection_test_loader(cfg, "ETIS_val")
evaluator = Trainer.build_evaluator(cfg, "ETIS_val")
print(inference_on_dataset(model, data_loader, evaluator))

print("\n###################### CVC_ClinicDB_val #################\n")
data_loader = build_detection_test_loader(cfg, "CVC_ClinicDB_val")
evaluator = Trainer.build_evaluator(cfg, "CVC_ClinicDB_val")
print(inference_on_dataset(model, data_loader, evaluator))

print("\n###################### CVC_ColonDB_val #################\n")
data_loader = build_detection_test_loader(cfg, "CVC_ColonDB_val")
evaluator = Trainer.build_evaluator(cfg, "CVC_ColonDB_val")
print(inference_on_dataset(model, data_loader, evaluator))

print("\n###################### CVC_300_val #################\n")
data_loader = build_detection_test_loader(cfg, "CVC_300_val")
evaluator = Trainer.build_evaluator(cfg, "CVC_300_val")
print(inference_on_dataset(model, data_loader, evaluator))

print("\n###################### Kvasir_val #################\n")
data_loader = build_detection_test_loader(cfg, "Kvasir_val")
evaluator = Trainer.build_evaluator(cfg, "Kvasir_val")
print(inference_on_dataset(model, data_loader, evaluator))



