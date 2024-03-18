import os
import pandas as pd
from torchvision.io import read_image
from torchvision.datasets import VisionDataset


class AgricultureVisionDataset(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.rgb_img_dir = self.root + "/images/rgb"
        self.nir_img_dir = self.root + "/images/nir"
        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
