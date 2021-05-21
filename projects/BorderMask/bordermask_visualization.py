
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
from tqdm import tqdm
# import mmcv
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.structures import Boxes, Instances
# from detectron2.projects.BorderMask import add_bordermask_config

from border_mask import add_bordermask_config


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
dataset_register()
add_bordermask_config(cfg)
cfg.merge_from_file("config/polyp/bordermask_LOG_R_101_FPN.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "../../polyp/bordermask_LOG_R_101_FPN/model_final.pth"
base_visPath = "../../polyp_eval/pred_fig/bordermask_LOG_R_101_FPN_0.5"

predictor = DefaultPredictor(cfg)

dataset_register()
model = predictor.model

####  #for each image in dataloader
# print("\n###################### ETIS_val #################\n")
# visPath = os.path.join(base_visPath,"ETIS_val")
# import os
# if not os.path.isdir(visPath):
#     os.makedirs(visPath)
# meta = MetadataCatalog.get("ETIS_val")
# data_loader = build_detection_test_loader(cfg, "ETIS_val")
#
# for idx, inputs in tqdm(enumerate(data_loader)):
#
#     file_name = inputs[0]['file_name']
#     # print(file_name)
#     img =  cv2.imread(file_name)
#     base_name = file_name.split("/")[-1]
#     outputs = model(inputs)[0]
#
#     v = Visualizer(img[:, :, ::-1],
#                    metadata=meta,
#                    instance_mode=ColorMode.IMAGE_BW
#                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
#                    )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#
#     cv2.imwrite(os.path.join(visPath, base_name), out.get_image()[:, :, ::-1])
#
# print("\n###################### CVC_ClinicDB_val #################\n")
# visPath = os.path.join(base_visPath,"CVC_ClinicDB_val")
# import os
# if not os.path.isdir(visPath):
#     os.makedirs(visPath)
# meta = MetadataCatalog.get("CVC_ClinicDB_val")
# data_loader = build_detection_test_loader(cfg, "CVC_ClinicDB_val")
# for idx, inputs in tqdm(enumerate(data_loader)):
#
#     file_name = inputs[0]['file_name']
#     # print(file_name)
#     img =  cv2.imread(file_name)
#     base_name = file_name.split("/")[-1]
#     outputs = model(inputs)[0]
#
#     v = Visualizer(img[:, :, ::-1],
#                    metadata=meta,
#                    instance_mode=ColorMode.IMAGE_BW
#                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
#                    )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#
#     cv2.imwrite(os.path.join(visPath, base_name), out.get_image()[:, :, ::-1])
#
# print("\n###################### CVC_ColonDB_val #################\n")
# visPath = os.path.join(base_visPath,"CVC_ColonDB_val")
# import os
# if not os.path.isdir(visPath):
#     os.makedirs(visPath)
# meta = MetadataCatalog.get("CVC_ColonDB_val")
# data_loader = build_detection_test_loader(cfg, "CVC_ColonDB_val")
# for idx, inputs in tqdm(enumerate(data_loader)):
#
#     file_name = inputs[0]['file_name']
#     # print(file_name)
#     img =  cv2.imread(file_name)
#     base_name = file_name.split("/")[-1]
#     outputs = model(inputs)[0]
#
#     v = Visualizer(img[:, :, ::-1],
#                    metadata=meta,
#                    instance_mode=ColorMode.IMAGE_BW
#                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
#                    )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#
#     cv2.imwrite(os.path.join(visPath, base_name), out.get_image()[:, :, ::-1])

print("\n###################### CVC_300_val #################\n")
visPath = os.path.join(base_visPath,"CVC_300_val")
import os
if not os.path.isdir(visPath):
    os.makedirs(visPath)
meta = MetadataCatalog.get("CVC_300_val")
data_loader = build_detection_test_loader(cfg, "CVC_300_val")
for idx, inputs in tqdm(enumerate(data_loader)):

    file_name = inputs[0]['file_name']
    # print(file_name)
    img =  cv2.imread(file_name)
    base_name = file_name.split("/")[-1]
    outputs = model(inputs)[0]

    v = Visualizer(img[:, :, ::-1],
                   metadata=meta,
                   instance_mode=ColorMode.IMAGE_BW
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(os.path.join(visPath, base_name), out.get_image()[:, :, ::-1])

# print("\n###################### Kvasir_val #################\n")
# visPath = os.path.join(base_visPath,"Kvasir_val")
# import os
# if not os.path.isdir(visPath):
#     os.makedirs(visPath)
# meta = MetadataCatalog.get("Kvasir_val")
# data_loader = build_detection_test_loader(cfg, "Kvasir_val")
# for idx, inputs in tqdm(enumerate(data_loader)):
#
#     file_name = inputs[0]['file_name']
#     # print(file_name)
#     img =  cv2.imread(file_name)
#     base_name = file_name.split("/")[-1]
#     outputs = model(inputs)[0]
#
#     v = Visualizer(img[:, :, ::-1],
#                    metadata=meta,
#                    instance_mode=ColorMode.IMAGE_BW
#                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
#                    )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#
#     cv2.imwrite(os.path.join(visPath, base_name), out.get_image()[:, :, ::-1])
