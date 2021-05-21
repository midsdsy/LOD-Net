# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import random
import cv2
from fvcore.transforms.transform import Transform
from albumentations import (MotionBlur,Blur)

class MotionBlurTransform(Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        prob,
        kernel_size
    ):
        super().__init__()
        self._set_attributes(locals())
        self.prob = prob
        self.kernel_size = kernel_size

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img):
        transform =MotionBlur(blur_limit=self.kernel_size, always_apply=False, p=self.prob)
        img = transform(image = img)['image']
        return img


