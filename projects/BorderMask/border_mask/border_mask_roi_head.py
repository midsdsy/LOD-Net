# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch

from detectron2.layers import ShapeSpec, cat, interpolate
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.mask_head import (
    build_mask_head,
    mask_rcnn_inference,
    mask_rcnn_loss,
)
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.modeling.poolers import ROIPooler

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from typing import Dict, List, Optional, Tuple
from detectron2.utils.events import get_event_storage
from torch.nn import functional as F

from .border_head import build_border_head
import logging

logger = logging.getLogger("Log_bordermask")


@ROI_HEADS_REGISTRY.register()
class BorderMaskROIHeads(StandardROIHeads):
    """
    The RoI heads class for PointRend instance segmentation models.

    In this class we redefine the mask head of `StandardROIHeads` leaving all other heads intact.
    To avoid namespace conflict with other heads we use names starting from `mask_` for all
    variables that correspond to the mask head in the class's namespace.
    """

    def __init__(self, cfg, input_shape):
        # TODO use explicit args style
        super().__init__(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES  ###something to verify
        self.mask_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self._feature_scales = {k: 1.0 / v.stride for k, v in input_shape.items()}

        # BorderMaskROIHeads source config
        self.pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        self.pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        self.sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        self.in_channels = [input_shape[f].channels for f in self.in_features][0]
        self.points_num = cfg.MODEL.BORDER_HEAD.POINTS_NUM

        # in_channels = np.sum([input_shape[f].channels for f in self.mask_coarse_in_features])

        if self.pooler_type:
            self.mask_pooler = ROIPooler(
                output_size=self.pooler_resolution,
                scales=self.pooler_scales,
                sampling_ratio=self.sampling_ratio,
                pooler_type=self.pooler_type,
            )
            shape = ShapeSpec(
                channels=self.in_channels, width=self.pooler_resolution, height=self.pooler_resolution
            )
        else:
            self.mask_pooler = None
            shape = {f: input_shape[f] for f in self.in_features}
        self.mask_rcnn_head = build_mask_head(cfg, shape)

        self.border_head = build_border_head(
            cfg,
            ShapeSpec(
                channels=self.in_channels,
                width=28,
                height=28,
            ),
        )
        self.parallel_border_head = build_border_head(
            cfg,
            ShapeSpec(
                channels=self.in_channels,
                width=28,
                height=28,
            ),
        )

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """

        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            # https://github.com/pytorch/pytorch/issues/41448
            features = dict([(f, features[f]) for f in self.mask_in_features])

        # border_mask_v2
        # gain the source pred of maskrcnn
        mask_logits = self._forward_mask_rcnn(features, instances)  # [N C 28 28]

        if self.training:
            # return loss_mask & aux_loss
            return self._forward_parallel_border(mask_logits, features, self.points_num, instances)
        else:
            # add "chosen_coordinate" attribute for instances, which is used for visualization
            chosen_coordinates, border_mask_logits = self._forward_parallel_border(mask_logits, features,
                                                                                   self.points_num, instances)
            # try:
            #     instances[0].chosenP = chosen_coordinates
            # except AssertionError:
            #     import pdb
            #     pdb.set_trace()
            # multi-scale test时，分配chosen_coordinates


            instance_num = 0
            for instance in instances:
                instance.chosenP = chosen_coordinates[instance_num: instance_num + len(instance), :, :]
                instance_num += len(instance)

            # v2_2.0
            mask_rcnn_inference(border_mask_logits,
                                instances)  # mask_rcnn_inference: add "pred_masks" attribute for instances

            return instances


    def _forward_mask_rcnn(self, features, boxs):
        return self.mask_rcnn_head(features, boxs)


    def _forward_parallel_border(self, mask_logits, features, points_num, instances):
        return self.parallel_border_head.forward(mask_logits, features, points_num, instances)
