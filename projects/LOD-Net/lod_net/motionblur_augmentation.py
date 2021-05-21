from detectron2.data.transforms.augmentation import Augmentation
from fvcore.transforms.transform import (
    NoOpTransform,
)
from .motionblur_transform import MotionBlurTransform
class MotionBlur(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob, kernel):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()
        self.prob = prob
        self.kernel_size=kernel
        self._init(locals())

    def get_transform(self, image):
        do = self._rand_range() < self.prob
        if do:
            return MotionBlurTransform(prob=self.prob, kernel_size=self.kernel_size)
        else:
            return NoOpTransform()