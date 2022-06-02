import math

from torchvision.transforms import Pad
from torchvision.transforms import functional as F


class CustomPad(Pad):
    """
    Represents a custom padding class compatible with torchvision.transforms. Pads a given image or tensor to a square
    format when no parameters are given. With parameters, it pads the image to square and then with the super to the
    intended padding.
    """

    def __init__(self, padding, fill=(255, 255, 255, 1), padding_mode="constant"):
        super().__init__(padding, fill, padding_mode)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Squared Padded image.
        """

        width = img.size[0]
        height = img.size[1]

        if width == height:
            _img = img

        if width < height:
            offset = height - width

            if offset % 2 == 0:  # even
                padding = int(offset / 2)
                _img = Pad(padding=[padding, 0, padding, 0], fill=(255, 255, 255, 1))(img)

            else:  # odd
                left = math.floor(offset / 2)
                right = offset - left
                _img = Pad(padding=[left, 0, right, 0], fill=(255, 255, 255, 1))(img)

        if width > height:
            offset = width - height

            if offset % 2 == 0:  # even
                padding = int(offset / 2)
                _img = Pad(padding=[0, padding, 0, padding], fill=(255, 255, 255, 1))(img)

            else:  # odd
                top = math.floor(offset / 2)
                bottom = offset - top
                _img = Pad(padding=[0, top, 0, bottom], fill=(255, 255, 255, 1))(img)

        return F.pad(_img, self.padding, self.fill, self.padding_mode)
