import os
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader


class AgricultureVisionDataset(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.rgb_img_dir = os.path.join(self.root, "images", "rgb")
        self.nir_img_dir = os.path.join(self.root, "images", "nir")
        self.img_labels = os.path.join(self.root, "labels")
        self.imgs = pd.read_csv(os.path.join(self.root, "data.csv"))

        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

        self.labels = (
            "double_plant",
            "nutrient_deficiency",
            "water",
            "drydown",
            "planter_skip",
            "waterway",
            "endrow",
            "storm_damage",
            "weed_cluster",
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.rgb_img_dir, self.imgs.iloc[idx, 0] + ".jpg")
        image = read_image(img_path)

        labels = []
        for lab in self.labels:
            label = os.path.join(self.img_labels, lab, self.imgs.iloc[idx, 0] + ".png")
            labels.append(read_image(label))
        labels = torch.stack([item.squeeze() for item in labels])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, labels


if __name__ == "__main__":
    dataset = AgricultureVisionDataset("./data/Agriculture-Vision-2021/train/")
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    imgs, labels = next(iter(data_loader))

    print("Batch of images has shape: ", imgs.shape)
    print("Batch of labels has shape: ", labels.shape)

    # visualize images
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4)
    for i in range(4):
        ax[i].imshow(imgs[i].permute(1, 2, 0))
    plt.show()
