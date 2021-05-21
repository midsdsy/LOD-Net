# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
import logging
from detectron2.utils.registry import Registry

BORDER_HEAD_REGISTRY = Registry("BORDER_HEAD")
BORDER_HEAD_REGISTRY.__doc__ = """
Registry for border heads, which *******************.

The registered object will be called with `obj(cfg, input_shape)`.
"""

@BORDER_HEAD_REGISTRY.register()
class BorderHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dim: the output dimension of the conv layers
            fc_dim: the feature dimenstion of the FC layers
            num_fc: the number of FC layers
            output_side_resolution: side resolution of the output square mask prediction
        """
        super(BorderHead, self).__init__()



    def forward(self, x):

        logger = logging.getLogger(__name__)
        logger.info("through into border_mask_head")
        logger.info("x.shape:{}".format(x.shape))

        return x

def build_border_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.ROI_BORDER_HEAD.NAME
    return BORDER_HEAD_REGISTRY.get(head_name)(cfg, input_channels)