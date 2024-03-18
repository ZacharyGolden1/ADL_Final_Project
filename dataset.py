import os
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
from torchvision.transforms import Grayscale
from torch.utils.data import DataLoader


class AgricultureVisionDataset(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.imgs = pd.read_csv(os.path.join(self.root, "data.csv"))
        self.rgb_img_dir = os.path.join(self.root, "images", "rgb")
        self.nir_img_dir = os.path.join(self.root, "images", "nir")
        self.img_labels = os.path.join(self.root, "labels")
        self.masks = os.path.join(self.root, "masks")

        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

        # downsample for baseline
        self.labels = (
            # "double_plant",
            # "nutrient_deficiency",
            # "water",
            # "drydown",
            # "planter_skip",
            "waterway",
            # "endrow",
            # "storm_damage",
            # "weed_cluster",
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

        mask_path = os.path.join(self.masks, self.imgs.iloc[idx, 0] + ".png")
        mask = read_image(mask_path)

        # mask image
        image = image * mask

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels


if __name__ == "__main__":
    target_transform = lambda x: Grayscale(1)(x) / 255.0
    dataset = AgricultureVisionDataset(
        "./data/Agriculture-Vision-2021/train/",
        transform=lambda x: (255.0 - x) / 255.0,
        target_transform=target_transform,
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    imgs, labels = next(iter(data_loader))

    print("Batch of images has shape: ", imgs.shape, " and type ", imgs.dtype)
    print("Batch of labels has shape: ", labels.shape, " and type ", labels.dtype)

    # visualize images
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4)
    for i in range(4):
        ax[0, i].imshow(imgs[i].permute(1, 2, 0))
        ax[1, i].imshow(labels[i].squeeze() * 255, cmap="gray")
    plt.show()

    for i in range(5):
        imgs, labels = next(iter(data_loader))
        print(torch.max(labels))
