import torch
# import some common libraries
import argparse
import os
import cv2
import matplotlib
import numpy
import torch
from lod_net import add_lod_config
from tqdm import tqdm
import random
from scipy import misc
import matplotlib.pyplot as plt

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

def position_convert_grids(grids, target_bbox):
    x0,y0,x1,y1 = target_bbox
    bbox_h = y1 - y0
    bbox_w = x1 - x0

    ori_h, ori_w,_ = grids.shape

    scale_h = bbox_h / ori_h
    scale_w = bbox_w / ori_w

    target_loc = grids.view(ori_h * ori_w, 2)
    target_loc[:, 0] = target_loc[:, 0] * scale_w + x0 +( bbox_w / ori_w / 2.)
    target_loc[:, 1] = target_loc[:, 1] * scale_h + y0 + (bbox_h / ori_h / 2.)

    # target_loc_x = target_loc[:, 0].data.cpu().numpy()
    # target_loc_y = target_loc[:, 1].data.cpu().numpy()
    # return target_loc_x, target_loc_y

    target_loc = target_loc.data.cpu().numpy()
    return target_loc


def position_convert_grids_offsets(grids_offsets, target_bbox):
    x0, y0, x1, y1 = target_bbox
    bbox_h = y1 - y0
    bbox_w = x1 - x0
    ori_h, ori_w, _, _ = grids_offsets.shape
    scale_h = bbox_h / ori_h
    scale_w = bbox_w / ori_w

    target_loc = grids_offsets.view(ori_h * ori_w, 2,8)
    target_loc = target_loc.permute(0, 2, 1)

    target_loc[:,:, 0] = target_loc[:,:, 0] * scale_w + x0 + (bbox_w / ori_w / 2.)
    target_loc[:,:, 1] = target_loc[:,:, 1] * scale_h + y0 + (bbox_h / ori_h / 2.)

    # target_loc_x = target_loc[:,:, 0].data.cpu().numpy()
    # target_loc_y = target_loc[:,:,1].data.cpu().numpy()
    # return target_loc_x, target_loc_y

    target_loc = target_loc.data.cpu().numpy()
    return target_loc



def main(args):
    cfg = get_cfg()
    datasets = dataset_register()
    add_lod_config(cfg)

    args.config = args.config if args.config is not None else "config/LOD_R_101_FPN_1x.yaml"
    args.weights = args.weights if args.weights is not None else "./outputs/LOD_R_101_FPN_1x_abs/model_final.pth"
    args.output_path = args.output_path if args.output_path is not None else "./outputs/vis_vectors/LOD_R_101_FPN_1x_abs"
    tensor_path = './outputs/vis_vectors/LOD_R_101_FPN_1x_abs/'
    cfg.merge_from_file(args.config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = args.weights
    base_path = args.output_path

    cfg.SOLVER.IMS_PER_BATCH = 1
    #multi-scale test
    # cfg.TEST.AUG.ENABLED =True

    predictor = DefaultPredictor(cfg)

    model = predictor.model
    if cfg.TEST.AUG.ENABLED:
        from detectron2.modeling import GeneralizedRCNNWithTTA

        model = GeneralizedRCNNWithTTA(cfg, model)

    # datasets = [ "CVC_ClinicDB_val"]
    for dataset in datasets:
        print("\n######################{}#################\n".format(dataset))
        visPath = os.path.join(base_path, dataset)
        if not os.path.isdir(visPath):
            os.makedirs(visPath)
        data_loader = build_detection_test_loader(cfg, dataset)
        for idx, inputs in tqdm(enumerate(data_loader)):
            file_name = inputs[0]['file_name']
            base_name = file_name.split("/")[-1]
            img = cv2.imread(file_name)
            with torch.no_grad():
                outputs = model(inputs)[0]


            grids_offsets = torch.load(tensor_path + 'grids_offsets.pt')  # [N 28 28 2 8]
            grids = torch.load(tensor_path + 'grids.pt')  # [28 28 2]
            bboxs = outputs['instances'].pred_boxes.tensor
            num_instances = grids_offsets.size(0)
            for idx_i in range(num_instances):

                # H,W,_ = img.shape
                bbox = bboxs[idx_i]
                x0, y0, x1, y1 = bbox
                grids_offset = grids_offsets[idx_i]

                if idx_i > 0:
                    grids = torch.load(tensor_path + 'grids.pt')

                # fig =plt.figure()
                # plt.imshow(img,cmap='autumn')
                # rect = plt.Rectangle((bbox[0], bbox[1]), bbox_w, bbox_h, fill=False, edgecolor='green', linewidth=2)
                # currentAxis = plt.gca()
                # currentAxis.add_patch(rect)

                origin_points = position_convert_grids(grids, bbox)
                sampled_points = position_convert_grids_offsets(grids_offset, bbox)


                #random draw
                res_img = cv2.rectangle(img, (x0, y0), (x1, y1), (0,0,255),1)
                for i in range(5):
                    idd = random.randint(0, 28 * 28 - 1)
                    origin_points_x, origin_points_y = origin_points[idd]
                    sampled_points_x, sampled_points_y = sampled_points[idd, :, 0], sampled_points[idd, :, 1]

                    for j in range(8):
                        dx = sampled_points_x[j]- origin_points_x
                        dy = sampled_points_y[j] - origin_points_y
                        # plt.arrow(origin_points_x, origin_points_y, dx, dy, width=0.5,head_length=3, head_width=3, fc='r', ec='r',linewidth=0.05, shape='full')
                        res_img = cv2.arrowedLine(res_img, (origin_points_x, origin_points_y), (sampled_points_x[j], sampled_points_y[j]), (0, 255, 0), 1, 0, 0, 0.3)
                    # plt.plot(sampled_points_x, sampled_points_y, 'o', color='red', linewidth=0.1,
                    #          markersize=2)
                    # plt.plot(origin_points_x, origin_points_y, 'o', color='yellow', linewidth=0.1, markersize=2)

            # plt.show()
            cv2.imwrite(os.path.join(visPath, base_name), res_img)
            # plt.savefig(os.path.join(visPath, base_name))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, metavar="FILE", help="path to config file")
    parser.add_argument("--weights", default=None, help="perform evaluation only")
    parser.add_argument("--output_path", default=None, help="number of gpus *per machine*")

    args = parser.parse_args()
    print(args)

    main(args)
















