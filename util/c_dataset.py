import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PapyriDataset(Dataset):
    """Papyri Cliplets dataset."""

    def __init__(self, root_path, transform=None):
        self.data_paths = [f"{root_path}/{f}" for f in sorted(os.listdir(root_path)) if f.endswith(".png")]
        self.label_char = [f.split("_")[0] for f in sorted(os.listdir(root_path)) if f.endswith(".png")]
        self.label_frag = [f.split("_")[1] for f in sorted(os.listdir(root_path)) if f.endswith(".png")]
        self.transform = transform

    def __getitem__(self, idx):
        # img = cv2.imread(self.data_paths[idx])
        with Image.open(self.data_paths[idx]) as img:
            _img = img.convert('RGB')

            if self.transform:
                _img = self.transform(img)

            label_char = self.label_char[idx]
            label_frag = self.label_frag[idx]

            return _img, label_char, label_frag

    def __len__(self):
        return len(self.data_paths)

    def __repr__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    TRAIN_PATH = "../data/raw-cleaned-custom"
    train_data = PapyriDataset(root_path=TRAIN_PATH, transform=transforms.ToTensor())
    print(train_data.__getitem__(0))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
