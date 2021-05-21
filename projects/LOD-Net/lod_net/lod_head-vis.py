# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec
from detectron2.utils.registry import Registry

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

import math
BORDER_HEAD_REGISTRY = Registry("BORDER_HEAD")
BORDER_HEAD_REGISTRY.__doc__ = ""

def build_lod_head(cfg, input_channels):
    """
    Build a border head defined by `cfg.MODEL.BORDER_HEAD.NAME`.
    """
    head_name = cfg.MODEL.BORDER_HEAD.NAME
    return BORDER_HEAD_REGISTRY.get(head_name)(cfg, input_channels)


@BORDER_HEAD_REGISTRY.register()
class LODBorderHead(nn.Module):
    """
    A head for Oriented derivatives learning, adaptive thresholding and feature fusion.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):

        super(LODBorderHead, self).__init__()
        self.input_channels = input_shape.channels
        self.num_directions = cfg.MODEL.ROI_MASK_HEAD.NUM_DIRECTIONS
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.conv_norm_relus = []
        self.cur_step = 0
        self.writer = SummaryWriter()

        cur_channels = cfg.MODEL.BORDER_HEAD.CHANNELS

        self.layers = nn.Sequential(
            Conv2d(
                self.input_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            ConvTranspose2d(
                cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
            ),
            nn.ReLU(),
            Conv2d(cur_channels, self.num_directions, kernel_size=3, padding=1, stride=1),
        )

        self.offsets_conv = nn.Sequential(
            Conv2d(
                self.input_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            ConvTranspose2d(
                cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
            ),
            nn.ReLU(),
            Conv2d(cur_channels, self.num_directions * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.num_directions * 2)
        )

        self.predictor = Conv2d(cur_channels, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.outputs_conv = Conv2d(self.num_directions, cur_channels, kernel_size=1)
        self.fusion_conv = nn.Sequential(
            Conv2d(
                self.input_channels + 1,
                cur_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                activation=nn.ReLU(),
            ),
            Conv2d(
                cur_channels,
                cur_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=nn.ReLU(),
            ),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)
            if isinstance(m, ConvTranspose2d):
                weight_init.c2_msra_fill(m)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # use normal distribution initialization for mask prediction layer
        weight_init.c2_msra_fill(self.outputs_conv)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, mask_logits, features, points_num, instances):

        pred_od = self.layers(features)
        offsets = self.offsets_conv(features)

        # torch.save(instances, 'instances.pt')
        # torch.save(features, 'features.pt')

        if self.training:
            gt_masks = self.cal_gt_masks(pred_od, instances)
            od_loss = self.oriented_derivative_learning(pred_od, offsets, gt_masks)
            losses = {"OD_loss": od_loss}

            od_activated_map, _ = self.adaptive_thresholding(points_num, pred_od)
            border_mask_logits = self.boundary_aware_mask_scoring(mask_logits, od_activated_map, pred_od)

            losses.update({"loss_mask": mask_rcnn_loss(border_mask_logits, instances)})
            return losses
        else:
            # self.for_visualization(pred_od, offsets)
            od_activated_map, _ = self.adaptive_thresholding(points_num, pred_od)
            border_mask_logits = self.boundary_aware_mask_scoring(mask_logits, od_activated_map, pred_od)
            return od_activated_map, border_mask_logits

    def for_visualization(self, pred_od, pred_offsets):

        N, C, H, W = pred_od.shape # [N 8 28 28]

        pred_offsets = pred_offsets.reshape(N, H, W, 2, 8)

        grids = self.mask_reference_point(H, W, device=pred_od.device)

        torch.save(grids, './outputs/vis_vectors_in_border/LOD_R_101_FPN_1x_abs/grids.pt')
        grids = grids[None, :, :, :, None]
        grids = grids.repeat(N, 1, 1, 1, C) #[N 28 28 2 8]

        grids_offsets = torch.tensor([[-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]],
                                     dtype=pred_offsets.dtype,
                                     device=pred_offsets.device)
        grids_offsets = grids_offsets.repeat(N, H, W, 1, 1)
        offsets = pred_offsets + grids_offsets
        grids = grids + offsets

        torch.save(grids, './outputs/vis_vectors_in_border/LOD_R_101_FPN_1x_abs/grids_offsets.pt')


    def boundary_aware_mask_scoring(self, mask_logits, od_activated_map, pred_od):

        od_features = self.outputs_conv(pred_od)
        od_activated_map = od_activated_map.unsqueeze(dim=1)

        mask_fusion_scores = mask_logits + od_features
        border_mask_scores = ~od_activated_map * mask_logits \
                                 + od_activated_map * mask_fusion_scores

        border_mask_scores = self.predictor(border_mask_scores)
        self.writer.close()
        return border_mask_scores


    def cal_gt_masks(self, preds, instances):

        cls_agnostic_mask = preds.size(1) == 1
        mask_side_len = preds.size(2)
        assert preds.size(2) == preds.size(3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device = preds.device)
            gt_masks.append(gt_masks_per_image)

        gt_masks = cat(gt_masks, dim=0)
        gt_masks = gt_masks.to(dtype=torch.float32)

        return gt_masks

    def mask_reference_point(self, H, W, device):
        ref_x, ref_y = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=device),
                                      torch.linspace(0, W - 1, W, dtype=torch.float32, device=device))
        ref = torch.cat((ref_x[..., None], ref_y[..., None]), -1)
        return ref

    def oriented_derivative_learning(self, features, pred_offsets, gt_masks):

        gt_masks = gt_masks.unsqueeze(1)
        N, C, H, W = gt_masks.shape

        pred_offsets = pred_offsets.reshape((N, H, W, 2, -1))
        direction_nums = features.size(1)
        features_after_sample = features.new_zeros((N, direction_nums, H, W))

        grids = self.mask_reference_point(H, W, device=features.device)
        grids = grids[None, :, :, :, None]
        grids = grids.repeat(N, 1, 1, 1, direction_nums)

        grids_offsets = torch.tensor([[-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]], dtype=pred_offsets.dtype,
                                     device=pred_offsets.device)
        grids_offsets = grids_offsets.repeat(N, H, W, 1, 1)
        offsets = pred_offsets + grids_offsets
        grids = grids + offsets

        inputs = gt_masks
        for num in range(direction_nums):
            per_direction_grids = grids[:, :, :, :, num]
            features_after_sample[:, num:num + 1, :, :] = F.grid_sample(inputs, per_direction_grids, mode='bilinear',
                                                                        align_corners=False, padding_mode='border')

        extend_gt_masks = gt_masks.repeat(1, direction_nums, 1, 1)

        # extend_gt_masks = abs(extend_gt_masks - features_after_sample)
        extend_gt_masks = extend_gt_masks - features_after_sample

        oriented_gt = torch.zeros_like(extend_gt_masks)

        for num in range(direction_nums):
            offset = offsets[:, :, :, :, num]
            dis = torch.rsqrt(torch.square(offset[:, :, :, 0]) + torch.square(offset[:, :, :, 1]) + torch.ones_like(
                offset[:, :, :, 0]))
            dis = dis[:, None, :, :]
            oriented_gt[:, num:num + 1, :, :] = (extend_gt_masks[:, num:num + 1, :, :] + 1).mul(dis)

        od_loss = F.smooth_l1_loss(features, oriented_gt, reduction="mean")

        return od_loss

    def adaptive_thresholding(self, points_num, od_features):

        N, C, H, W = od_features.shape
        # od_features = od_features.abs()

        # for each channel, choose top points_num points and add through channel
        oriented_activated_features = torch.zeros([N, H * W], dtype=torch.float32, device=od_features.device)
        for k in range(8):
            val, idx = torch.topk(od_features.view(N, C, H * W)[:, k, :], points_num)
            for i in range(N):
                oriented_activated_features[i, idx[i]] += od_features.view(N, C, H * W)[i, k, idx[i]]

        _, idxx = torch.topk(oriented_activated_features, points_num)
        shift = H * W * torch.arange(N, dtype=torch.long, device=idxx.device)
        idxx += shift[:, None]
        activated_map = torch.zeros([N, H * W], dtype=torch.bool, device=od_features.device)
        activated_map.view(-1)[idxx.view(-1)] = True

        return activated_map.view(N, H, W), oriented_activated_features.view(N, H, W)


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)



@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    # vis_period = 1
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            # import pdb
            # pdb.set_trace()
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss
