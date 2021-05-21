# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
import math
from detectron2.utils.events import get_event_storage
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec
import logging
from .border_head import BORDER_HEAD_REGISTRY

logger = logging.getLogger("Log_bordermask")


@BORDER_HEAD_REGISTRY.register()
class ParallelBorderHead(nn.Module):
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
        super(ParallelBorderHead, self).__init__()
        self.input_channels = input_shape.channels
        self.num_classes = 80

        # layers
        self.conv_norm_relus = []

        cur_channels = 256

        self.mask_fcn1 = Conv2d(
            self.input_channels,
            cur_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )
        self.mask_fcn2 = Conv2d(
            cur_channels,
            cur_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )
        self.mask_fcn3 = Conv2d(
            cur_channels,
            cur_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )
        self.mask_fcn4 = Conv2d(
            cur_channels,
            cur_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

        self.deconv = ConvTranspose2d(
            cur_channels, cur_channels, kernel_size=2, stride=2, padding=0
        )
        self.deconv_relu = nn.ReLU()
        self.transform_conv = Conv2d(cur_channels, 8, kernel_size=3, padding=1, stride=1)

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
            Conv2d(cur_channels, 8 * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(8 * 2)
        )

        # v2_2.0###
        self.predictor = Conv2d(cur_channels, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.outputs_conv = Conv2d(8, cur_channels, kernel_size=1)
        # use normal distribution initialization for mask prediction layer
        weight_init.c2_msra_fill(self.outputs_conv)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

        # v2_2.0###
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)
            if isinstance(m, ConvTranspose2d):
                weight_init.c2_msra_fill(m)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #
        # weight_init.c2_msra_fill(self.mask_fcn1)
        # weight_init.c2_msra_fill(self.mask_fcn2)
        # weight_init.c2_msra_fill(self.mask_fcn3)
        # weight_init.c2_msra_fill(self.mask_fcn4)
        # weight_init.c2_msra_fill(self.deconv)
        # weight_init.c2_msra_fill(self.transform_conv)

    def layers(self, x):
        x = self.mask_fcn1(x)
        x = self.mask_fcn2(x)
        x = self.mask_fcn3(x)
        x = self.mask_fcn4(x)
        x = self.deconv(x)
        x = self.deconv_relu(x)
        x = self.transform_conv(x)
        return x

    def forward(self, mask_logits, features, points_num, instances):
        '''

        Args:
            mask_logits:  from mask_head in mask-rcnn #[N C 28 28]
            features:  instances features
            points_num:
            instances:

        Returns:

        '''

        x = self.layers(features)  # x:[N 8 28 28]
        # offsets
        offsets = self.offsets_conv(features) # x:[N 16 28 28]
        
        deformable_features = self.sample_point_with_offset(offsets,x)

        # x = self.deconv(features)
        # x = self.transform_conv(x)

        if self.training:

            ############v6
            gt_masks = self.cal_gt_masks(x, instances)
            
            aux_loss = self.border_aux_loss(deformable_features, gt_masks)

            # choose top points_num points as the area of boundary. chosen_coordinate is a boolean martix with a same size of [N 28 28]
            chosen_coordinate = self.get_chosen_coordinate(points_num, deformable_features)
            

            losses = {"aux_loss": aux_loss}

            # v2_1.0
            # losses.update(mask_rcnn_loss_border(mask_logits, instances, points_num, chosen_coordinate))

            # v2_2.0
            # fusion loss
            border_mask_logits = self.border_cross(mask_logits, chosen_coordinate, deformable_features)

            losses.update({"loss_mask": mask_rcnn_loss(border_mask_logits, instances)})
            return losses
        else:

            # v1 & v2_1.0
            chosen_coordinate = self.get_chosen_coordinate(points_num, deformable_features)
            # v2_2.0
            border_mask_logits = self.border_cross(mask_logits, chosen_coordinate, deformable_features)

            return chosen_coordinate, border_mask_logits
        
    def sample_point_with_offset(self, offsets, features):
        N,C,H,W = offsets.shape
        offsets = offsets.reshape((N, H, W, 2, -1)) #[N 28 28 2 8]
       
        direction_nums = features.size(1)

        features_after_sample = features.new_zeros((N, 8, H, W))

        grids = self.mask_reference_point(H, W, device=features.device) #[28 28 2]

        grids = grids[None, :, :, :, None]#[1 28 28 2 1]
        grids = grids.repeat(N, 1, 1, 1, 8) #[N 28 28 2 8]
        grids = grids + offsets

        for num in range(direction_nums):
            inputs = features[:, num:num + 1, :, :] #inputs[N 1 28 28]
            per_direction_grids = grids[:, :, :, :, num] #per_direction_grids[N 28 28 2]

            features_after_sample[:, num:num + 1, :, :] = F.grid_sample(inputs, per_direction_grids, mode='bilinear',
                                                                        align_corners=False, padding_mode='border')
        return features_after_sample

    def border_cross(self, mask_logits, chosen_coordinate, outputs_features):
        '''
        v2_2.0 require
        Args:
            mask_logits: [B 256 28 28]
            chosen_coordinate: [B 28 28] bool
            outputs_features: [B 8 28 28]

        Returns:
            border_mask_scores:[B 80 28 28]

        '''
        # import pdb
        # pdb.set_trace()
        outputs_features = self.outputs_conv(outputs_features)  # [B 8 28 28]
        mask_fusion_scores = mask_logits + outputs_features
        chosen_coordinate = chosen_coordinate.unsqueeze(dim=1)

        border_mask_scores = ~chosen_coordinate * mask_logits \
                             + chosen_coordinate * mask_fusion_scores

        border_mask_scores = self.predictor(border_mask_scores)

        return border_mask_scores

    def cal_gt_masks(self, pred_mask_logits, instances):
        ##copy from mask_rcnn_loss function

        cls_agnostic_mask = pred_mask_logits.size(1) == 1
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
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        gt_masks = cat(gt_masks, dim=0)
        gt_masks = gt_masks.to(dtype=torch.float32)

        return gt_masks

    def mask_reference_point(self, H, W, device):
        ref_x, ref_y = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.float32, device=device),
                                      torch.linspace(0, W - 1, W, dtype=torch.float32, device=device))
        ref = torch.cat((ref_x[..., None], ref_y[..., None]), -1)
        return ref
    def border_aux_loss(self, features, gt_masks):
        '''

        Args:
            features: [N 8 28 28]
            gt_masks: [N 28 28]
        Returns:

        '''
        gt_masks = gt_masks.unsqueeze(1) #[N 1 28 28]
        N,C,H,W = gt_masks.shape

        #copy gt_masks, gain extend_gt_masks with size of [N 8 28 28]
        extend_gt_masks = torch.cat([gt_masks, gt_masks, gt_masks, gt_masks, gt_masks, gt_masks, gt_masks, gt_masks], dim=1)
        n,c,h,w = extend_gt_masks.shape


        '''
            zhe li shi xian cuo wei, hou xiang jian.
            suo yi ,bi ru, dui yu extend_gt_masks de shang fang xiang, cuo wei shi xia yi, hou yong gt_masks jian qu extend_gt_masks, de dao ding yi xia de shang fang xiang cha zhi.
        '''
        #Dislocation / cuo wei
        #top
        extend_gt_masks[:, 0:C , 1:, :] = gt_masks[:, :, 0:h-1, :]
        #bottom
        extend_gt_masks[:, C:2*C, :h-1,:] = gt_masks[:, :, 1:,:]
        #left
        extend_gt_masks[:, 2*C:3*C, :, 1:] = gt_masks[:, :, :, :w - 1]
        #right
        extend_gt_masks[:, 3*C:4*C, :, :w - 1] = gt_masks[:, :, :, 1:]
        #top left
        extend_gt_masks[:, 4*C:5*C, 1:, 1:] = gt_masks[:, :, :h - 1, :w - 1]
        #top right
        extend_gt_masks[:, 5*C:6*C, 1:, :w - 1] = gt_masks[:, :, :h - 1, 1:]
        #bottom left
        extend_gt_masks[:, 6*C:7*C, :h - 1, 1:] = gt_masks[:, :, 1:, :w - 1]
        #bottom right
        extend_gt_masks[:, 7*C:8*C, :h - 1, :w - 1] = gt_masks[:, :, 1:, 1:]

        # non normalization
        ## better performance without normalization
        for k in range(8):
            #subtract
            extend_gt_masks[:, k * C:(k + 1) * C, :, :] = gt_masks - extend_gt_masks[:, k * C:(k + 1) * C, :, :]
            if k > 3:
                extend_gt_masks[:, k * C:(k + 1) * C, :, :] /= math.sqrt(2.0)

        # #normalization
        # maxx = gt_masks.view(-1, C * H * W).max(dim = 1)
        # minn = gt_masks.view(-1, C * H * W).min(dim = 1)
        # gap = maxx.values - minn.values
        #
        # for k in range(8):
        #     extend_gt_masks[:, k * C:(k + 1) * C, :, :] = gt_masks - extend_gt_masks[:, k * C:(k + 1) * C, :, :]
        #     if k > 3:
        #         extend_gt_masks[:, k * C:(k + 1) * C, :, :] /= math.sqrt(2.0)
        #
        # extend_gt_masks = ((extend_gt_masks.view(-1, 8 * C * H * W) + 0.00001 )/ (gap[:, None] + 0.00001)).view(N, 8 * C, H,
        #                                                                                                     W)


        # for k in range(8):
        #     extend_gt_masks[:, k * C:(k + 1) * C, :, :] = extend_gt_masks[:,k*C:(k+1)*C,:,:] + gt_masks
        #     if k<4:
        #         extend_gt_masks[:,k*C:(k+1)*C,:,:] = extend_gt_masks[:,k*C:(k+1)*C,:,:]/2.0
        #     else:
        #         extend_gt_masks[:, k * C:(k + 1) * C, :, :] = extend_gt_masks[:, k * C:(k + 1) * C, :, :] / (2.0/math.sqrt(2))
        #     extend_gt_masks[:,k*C:(k+1)*C,:,:] = (gt_masks - extend_gt_masks[:,k*C:(k+1)*C,:,:])
        #
        # extend_gt_masks = (extend_gt_masks.view(-1,8*C*H*W)+ 0.00001/gap[:,None]+ 0.00001).view( N, 8*C, H, W )

        aux_loss = F.smooth_l1_loss(features, extend_gt_masks, reduction="mean")
        # aux_loss = F.mse_loss(features, extend_gt_masks, reduction="mean")

        return aux_loss

    def border_derformable_aux_loss(self, features, offsets, gt_masks):
        '''

        Args:
            features: [N 8 28 28]
            offsets:[N 16 28 28]
            gt_masks: [N 28 28]
        Returns:

        '''

        gt_masks = gt_masks.unsqueeze(1)  # [N 1 28 28]
        N, C, H, W = gt_masks.shape

       

        # copy gt_masks, gain extend_gt_masks with size of [N 8 28 28]
        extend_gt_masks = torch.cat([gt_masks, gt_masks, gt_masks, gt_masks, gt_masks, gt_masks, gt_masks, gt_masks],
                                    dim=1)
        n, c, h, w = extend_gt_masks.shape

        '''
            zhe li shi xian cuo wei, hou xiang jian.
            suo yi ,bi ru, dui yu extend_gt_masks de shang fang xiang, cuo wei shi xia yi, hou yong gt_masks jian qu extend_gt_masks, de dao ding yi xia de shang fang xiang cha zhi.
        '''
        # Dislocation / cuo wei
        # top
        extend_gt_masks[:, 0:C, 1:, :] = gt_masks[:, :, 0:h - 1, :]
        # bottom
        extend_gt_masks[:, C:2 * C, :h - 1, :] = gt_masks[:, :, 1:, :]
        # left
        extend_gt_masks[:, 2 * C:3 * C, :, 1:] = gt_masks[:, :, :, :w - 1]
        # right
        extend_gt_masks[:, 3 * C:4 * C, :, :w - 1] = gt_masks[:, :, :, 1:]
        # top left
        extend_gt_masks[:, 4 * C:5 * C, 1:, 1:] = gt_masks[:, :, :h - 1, :w - 1]
        # top right
        extend_gt_masks[:, 5 * C:6 * C, 1:, :w - 1] = gt_masks[:, :, :h - 1, 1:]
        # bottom left
        extend_gt_masks[:, 6 * C:7 * C, :h - 1, 1:] = gt_masks[:, :, 1:, :w - 1]
        # bottom right
        extend_gt_masks[:, 7 * C:8 * C, :h - 1, :w - 1] = gt_masks[:, :, 1:, 1:]

        # non normalization
        ## better performance without normalization
        for k in range(8):
            # subtract
            extend_gt_masks[:, k * C:(k + 1) * C, :, :] = gt_masks - extend_gt_masks[:, k * C:(k + 1) * C, :, :]
            if k > 3:
                extend_gt_masks[:, k * C:(k + 1) * C, :, :] /= math.sqrt(2.0)

        # #normalization
        # maxx = gt_masks.view(-1, C * H * W).max(dim = 1)
        # minn = gt_masks.view(-1, C * H * W).min(dim = 1)
        # gap = maxx.values - minn.values
        #
        # for k in range(8):
        #     extend_gt_masks[:, k * C:(k + 1) * C, :, :] = gt_masks - extend_gt_masks[:, k * C:(k + 1) * C, :, :]
        #     if k > 3:
        #         extend_gt_masks[:, k * C:(k + 1) * C, :, :] /= math.sqrt(2.0)
        #
        # extend_gt_masks = ((extend_gt_masks.view(-1, 8 * C * H * W) + 0.00001 )/ (gap[:, None] + 0.00001)).view(N, 8 * C, H,
        #                                                                                                     W)

        # for k in range(8):
        #     extend_gt_masks[:, k * C:(k + 1) * C, :, :] = extend_gt_masks[:,k*C:(k+1)*C,:,:] + gt_masks
        #     if k<4:
        #         extend_gt_masks[:,k*C:(k+1)*C,:,:] = extend_gt_masks[:,k*C:(k+1)*C,:,:]/2.0
        #     else:
        #         extend_gt_masks[:, k * C:(k + 1) * C, :, :] = extend_gt_masks[:, k * C:(k + 1) * C, :, :] / (2.0/math.sqrt(2))
        #     extend_gt_masks[:,k*C:(k+1)*C,:,:] = (gt_masks - extend_gt_masks[:,k*C:(k+1)*C,:,:])
        #
        # extend_gt_masks = (extend_gt_masks.view(-1,8*C*H*W)+ 0.00001/gap[:,None]+ 0.00001).view( N, 8*C, H, W )

        aux_loss = F.smooth_l1_loss(features_after_sample, extend_gt_masks, reduction="mean")
        # aux_loss = F.mse_loss(features, extend_gt_masks, reduction="mean")

        return aux_loss

    def get_chosen_coordinate(self, points_num, features):

        N, C, H, W = features.shape  # [instance_num 8 28 28]
        sum_features = torch.zeros([N, H * W], dtype=torch.float32, device=features.device)

        # for each channel, choose top points_num points and add through channel
        for k in range(8):
            val, idx = torch.topk(features.view(N, C, H * W)[:, k, :], points_num)
            for i in range(N):
                sum_features[i, idx[i]] += features.view(N, C, H * W)[i, k, idx[i]]

        # for sum_features , choose top points_num points and record chosen_coordinate
        _, idxx = torch.topk(sum_features, points_num)
        shift = H * W * torch.arange(N, dtype=torch.long, device=idxx.device)
        idxx += shift[:, None]
        chosen_coordinate = torch.zeros([N, H * W], dtype=torch.bool, device=features.device)
        chosen_coordinate.view(-1)[idxx.view(-1)] = True

        return chosen_coordinate.view(N, H, W)


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


@torch.jit.unused
def mask_rcnn_loss_border(pred_mask_logits: torch.Tensor, instances: List[Instances], points_num,
                          chosen_coordinate: torch.Tensor, vis_period: int = 0):
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
        chosen_coordinate(torch.Tensor): shape(B,H*W),where B is the total number of predicted masks
            in all images
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
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
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
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    return BCE_with_coordinate(pred_mask_logits, gt_masks, chosen_coordinate, points_num)


def BCE_with_coordinate(pred_mask_logits, gt_masks, chosen_coordinate, points_num):
    # N,H,W = gt_masks.shape
    #
    # shift = H * W * torch.arange(N, dtype=torch.long, device=chosen_coordinate.device)
    #
    # chosen_coordinate += shift[:, None]
    # not_chosen_coordinate += shift[:, None]
    # pred_mask_logits_hard = pred_mask_logits.view(-1)[chosen_coordinate.view(-1)].view(N, points_num)
    # pred_mask_logits_easy = pred_mask_logits.view(-1)[not_chosen_coordinate.view(-1)].view(N, H * W - points_num)
    # gt_masks_hard = gt_masks.view(-1)[chosen_coordinate.view(-1)].view(N, points_num)
    # gt_masks_easy = gt_masks.view(-1)[not_chosen_coordinate.view(-1)].view(N, H * W - points_num)
    #
    #
    # mask_loss_hard = F.binary_cross_entropy_with_logits(pred_mask_logits_hard, gt_masks_hard, reduction="mean")
    # mask_loss_easy = F.binary_cross_entropy_with_logits(pred_mask_logits_easy, gt_masks_easy, reduction="mean")
    # loss_mask = mask_loss_hard+mask_loss_easy
    # return {
    #     "loss_mask":loss_mask
    #     }
    mask_loss_hard = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean",
                                                        weight=chosen_coordinate)
    mask_loss_easy = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean",
                                                        weight=~chosen_coordinate)
    # loss_mask = mask_loss_hard + mask_loss_easy
    return {
        "loss_mask_hard": mask_loss_hard,
        "loss_mask_easy": mask_loss_easy
    }


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
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
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
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    # import pdb
    # pdb.set_trace()
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss
