"""
Datasets and DataLoaders for the Agriculture Vision dataset.
"""

import os
import pandas as pd
import torch
from typing import Optional, Callable
from torchvision.io import read_image, ImageReadMode
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class AgricultureVisionDataset(VisionDataset):
    """
    Agriculture Vision dataset.

    Items:
        - image : H * W * 4 (RGB + NIR)
        - label : H * W * C (C classes based on labels selected)
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,  # transform image, label pair
        transform: Optional[Callable] = None,  # transform image
        target_transform: Optional[Callable] = None,  # transform label
    ):
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

        mask_path = os.path.join(self.masks, self.imgs.iloc[idx, 0] + ".png")
        mask = read_image(mask_path, ImageReadMode.GRAY)

        # mask image
        image = image * mask

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    dataset_size: Optional[int] = None,
):
    """
    Create dataloaders for the Agriculture Vision dataset.

    # TODO: different dataset sizes for train and test

    Returns: train_loader, test_loader
    """

    train_dataset = AgricultureVisionDataset(
        train_dir, transform=transform, target_transform=target_transform
    )
    if dataset_size:
        train_dataset = torch.utils.data.Subset(train_dataset, range(0, dataset_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,  # drop last batch if it is not full
    )

    test_dataset = AgricultureVisionDataset(
        test_dir, transform=transform, target_transform=target_transform
    )
    if dataset_size:
        test_dataset = torch.utils.data.Subset(test_dataset, range(0, dataset_size))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    invert_transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.functional.invert]
    )
    clamp_target_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            lambda x: torch.clamp(x, 0.0, 1.0),
        ]
    )
    dataset = AgricultureVisionDataset(
        "./data/Agriculture-Vision-2021/train/",
        transform=invert_transform,
        target_transform=clamp_target_transform,
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
        imgs, ls = next(iter(data_loader))
        print(torch.max(ls))
