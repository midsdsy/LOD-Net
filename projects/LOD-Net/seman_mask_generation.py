# import some common libraries
import argparse
import os
import cv2
import matplotlib
import torch
from lod_net import add_lod_config
from tqdm import tqdm
from scipy import misc

from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
)
from detectron2.engine import DefaultPredictor


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

    return [ "CVC_ClinicDB_val","ETIS_val", "Kvasir_val", "CVC_ColonDB_val", "CVC_300_val"]


def main(args):
    cfg = get_cfg()
    datasets = dataset_register()
    add_lod_config(cfg)

    args.config = args.config if args.config is not None else "./outputs/LOD_R_101_RPN_1x_p100_abs_intop/config.yaml"
    args.weights = args.weights if args.weights is not None else "./outputs/LOD_R_101_RPN_1x_p100_abs_intop/model_final.pth"
    args.output_path = args.output_path if args.output_path is not None else "./outputs/seman/LOD_R_101_RPN_1x_p100_abs_intop/"
    cfg.merge_from_file(args.config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = args.weights
    base_path = args.output_path

    cfg.SOLVER.IMS_PER_BATCH = 1
    #multi-scale test
    cfg.TEST.AUG.ENABLED =True

    predictor = DefaultPredictor(cfg)

    model = predictor.model
    if cfg.TEST.AUG.ENABLED:
        from detectron2.modeling import GeneralizedRCNNWithTTA

        model = GeneralizedRCNNWithTTA(cfg, model)

    for dataset in datasets:
        print("\n######################{}#################\n".format(dataset))
        visPath = os.path.join(base_path, dataset)
        if not os.path.isdir(visPath):
            os.makedirs(visPath)
        data_loader = build_detection_test_loader(cfg, dataset)
        for idx, inputs in tqdm(enumerate(data_loader)):
            file_name = inputs[0]['file_name']
            base_name = file_name.split("/")[-1]

            with torch.no_grad():
                outputs = model(inputs)[0]

            seman_mask = outputs['instances'].pred_masks
            seman_mask = seman_mask.sum(dim=0)
            seman_mask = (seman_mask >= 1) + 0

            matplotlib.image.imsave(os.path.join(visPath, base_name), seman_mask.cpu().numpy(), cmap='gray')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, metavar="FILE", help="path to config file")
    parser.add_argument("--weights", default=None, help="perform evaluation only")
    parser.add_argument("--output_path", default=None, help="number of gpus *per machine*")

    args = parser.parse_args()
    print(args)

    main(args)
