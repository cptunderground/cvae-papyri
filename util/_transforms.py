import math

import PIL.Image as Image
from torchvision.transforms import Pad
from torchvision.transforms import functional as F


class _Pad(Pad):
    def __init__(self, padding, fill=(255,255,255), padding_mode="constant"):
        super().__init__(padding, fill, padding_mode)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """

        width = img.size[0]
        height = img.size[1]

        if width == height:
            _img = img

        if width < height:
            offset = height - width
            if offset % 2 == 0:  # even
                padding = int(offset / 2)
                _img = Pad(padding=[padding, 0, padding, 0],fill=(255,255,255))(img)

            else:  # odd
                left = math.floor(offset / 2)
                right = offset - left
                _img = Pad(padding=[left, 0, right, 0],fill=(255,255,255))(img)

        if width > height:
            offset = width - height
            if offset % 2 == 0:  # even
                padding = int(offset / 2)
                _img = Pad(padding=[0, padding, 0, padding],fill=(255,255,255))(img)

            else:  # odd
                top = math.floor(offset / 2)
                bottom = offset - top
                _img = Pad(padding=[0, top, 0, bottom],fill=(255,255,255))(img)

        return F.pad(_img, self.padding, self.fill, self.padding_mode)
