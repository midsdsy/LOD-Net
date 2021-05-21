# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
import logging
import math
from detectron2.utils.registry import Registry

BORDER_HEAD_REGISTRY = Registry("BORDER_HEAD")
BORDER_HEAD_REGISTRY.__doc__ = """
Registry for border heads, which *******************.

The registered object will be called with `obj(cfg, input_shape)`.
"""
logger = logging.getLogger("Log_bordermask")

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
        self.input_channels = input_shape.channels
        self.num_classes = 80

        #layers
        self.conv1 = Conv2d(self.input_channels*9,self.input_channels,kernel_size=3, padding = 1 )
        self.predictor = Conv2d(self.input_channels, self.num_classes, kernel_size=1, stride=1, padding=0)

        # for layer in self.conv_norm_relus + [self.deconv]:
        #     weight_init.c2_msra_fill(layer)

        nn.init.normal_(self.predictor.weight, std=0.001)
        weight_init.c2_msra_fill(self.conv1)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)



    def forward(self, x):
        logger.info("through into border_head")
        logger.info("x.shape:{}".format(x.shape))

        N,C,H,W = x.shape
        direc = torch.cat([x,x,x,x,x,x,x,x],dim =1)
        n,c,h,w = direc.shape

        #top
        direc[:,0:C,1:,:] = x[:,:,0:h-1,:]
        #bottom
        direc[:, C:2*C, :h-1,:] = x[:, :, 1:,:]
        #left
        direc[:, 2*C:3*C, :, 1:] = x[:, :,:, :w - 1]
        #right
        direc[:, 3*C:4*C, :, :w - 1] = x[:, :, :, 1:]
        #top left
        direc[:, 4*C:5*C, 1:, 1:] = x[:, :, :h - 1, :w - 1]
        #top right
        direc[:, 5*C:6*C, 1:, :w - 1] = x[:, :, :h - 1, 1:]
        #bottom left
        direc[:, 6*C:7*C, :h - 1, 1:] = x[:, :, 1:, :w - 1]
        #bottom right
        direc[:, 7*C:8*C, :h - 1, :w - 1] = x[:, :, 1:, 1:]

        # maxx = x.view(-1, C * H * W).max(dim=1)
        # minn = x.view(-1, C * H * W).min(dim=1)
        # gap = maxx.values - minn.values
        #
        # for k in range(8):
        #     direc[:, k * C:(k + 1) * C, :, :] = x - direc[:, k * C:(k + 1) * C, :, :]
        #     if k > 3:
        #         direc[:, k * C:(k + 1) * C, :, :] /= math.sqrt(2.0)
        #
        # direc = ((direc.view(-1, 8 * C * H * W) + 0.00001) / (gap[:, None] + 0.00001)).view(N,8 * C,H,W)

        for k in range(8):
            direc[:, k * C:(k + 1) * C, :, :] = x - direc[:, k * C:(k + 1) * C, :, :]
            if k > 3:
                direc[:, k * C:(k + 1) * C, :, :] /= math.sqrt(2.0)
        # for k in range(8):
        #     direc[:, k * C:(k + 1) * C, :, :] = direc[:,k*C:(k+1)*C,:,:] + x
        #     direc[:,k*C:(k+1)*C,:,:] = direc[:,k*C:(k+1)*C,:,:] / 2.0
        #     direc[:,k*C:(k+1)*C,:,:] = (x - direc[:,k*C:(k+1)*C,:,:])

        x = torch.cat([x,direc],dim =1)
        x = self.conv1(x)

        # import pdb
        # pdb.set_trace()

        x = self.predictor(x)

        return x

def build_border_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.BORDER_HEAD.NAME
    return BORDER_HEAD_REGISTRY.get(head_name)(cfg, input_channels)