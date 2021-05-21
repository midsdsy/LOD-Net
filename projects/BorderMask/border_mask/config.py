
from detectron2.config import CfgNode as CN


def add_bordermask_config(cfg):
    """
    Add config for BorderMask.
    """
    cfg.MODEL.BORDER_HEAD = CN()
    cfg.MODEL.BORDER_HEAD.NAME = "BorderHead"
    cfg.MODEL.BORDER_HEAD.POINTS_NUM = 100


    # Names of the input feature maps to be used by a coarse mask head.
    cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.ROI_MASK_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_MASK_HEAD.NUM_FC = 2
    # The side size of a coarse mask head prediction.
    cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION = 7
    cfg.MODEL.ROI_MASK_HEAD.LEARNABLE_BOUNDARY = True
    cfg.MODEL.ROI_MASK_HEAD.NUM_DIRECTIONS = 8
    cfg.MODEL.ROI_MASK_HEAD.FUSION = True
    # ['avg', 'max', 'direction']
    cfg.MODEL.ROI_MASK_HEAD.MINING_TYPE = "direction"

    cfg.INPUT.AUG = False
    cfg.INPUT.BLUR = CN({"ENABLED": False})
    cfg.INPUT.BLUR.Prob = 0.5
    cfg.INPUT.BLUR.KERNEL_SIZE = 3

