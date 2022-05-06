import math

import PIL.Image
from torchvision.transforms import Pad
from torchvision.transforms import functional as F
class _Pad(Pad):
    def __init__(self,padding, fill=0, padding_mode="constant"):
        # ifchecks for square.
        super().__init__(padding, fill, padding_mode)

    def forward(self, img:PIL.Image.Image):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """

        print(img.size)
        print(img.width)
        print(img.height)
        width = img.width
        height = img.height

        if width == height:
            _img = img

        if width < height:
            offset = height - width
            if offset % 2 == 0: #even
                padding = offset /2
                _img = Pad(padding=[padding, 0 ,padding,0])(img)
                print(f"padded img size={_img.size}")
            else: #odd
                left = math.floor(offset/2)
                right = offset -left
                _img = Pad(padding=[left, 0 ,right,0])(img)
                print(f"padded img size={_img.size}")


        if width > height:
            offset = width - height
            if offset % 2 == 0:  # even
                padding = offset / 2
                _img = Pad(padding=[0, padding, 0, padding])(img)
                print(f"padded img size={_img.size}")
            else:  # odd
                top = math.floor(offset / 2)
                bottom = offset - top
                _img = Pad(padding=[0, top, 0, bottom])(img)
                print(f"padded img size={_img.size}")

        return F.pad(_img, self.padding, self.fill, self.padding_mode)

