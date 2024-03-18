import os
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader

ROOT = "/Users/golden/Desktop/CMU/Y4/Spring24/ADL/Final Project/Project Code/ood/data/Agriculture-Vision-2021/train/"
CSV = "/Users/golden/Desktop/CMU/Y4/Spring24/ADL/Final Project/Project Code/ood/data/Agriculture-Vision-2021/train/train.csv"

class AgricultureVisionDataset(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.rgb_img_dir = self.root + "images/rgb"
        self.nir_img_dir = self.root + "images/nir"
        self.img_labels = self.root + "labels"
        self.data_frame = pd.read_csv(CSV)

        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

        self.img_dim = (512, 512)
        self.labels = "double_plant", "nutrient_deficiency", "water", "drydown", \
                     "planter_skip","waterway","endrow","storm_damage", "weed_cluster"

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.rgb_img_dir, self.data_frame.iloc[idx, 0] + ".jpg")
        image = read_image(img_path)
        labels = []
        for lab in self.labels:
            label = os.path.join(self.img_labels, lab, self.data_frame.iloc[idx, 0] + ".png")
            labels.append(read_image(label))
        labels = torch.stack([item.squeeze() for item in labels])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, labels
    
    def get_batch(self):
        raise NotImplementedError


if __name__ == "__main__":
	dataset = AgricultureVisionDataset(root=ROOT)		
	data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
	for imgs, labels in data_loader:
		print("Batch of images has shape: ",imgs.shape)
		print("Batch of labels has shape: ", labels.shape)