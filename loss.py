"""
Loss and accuracy metric implementations
"""

import torch
from torch import nn


def weighted_l1_loss(pred, labels):
    # ratio of images with non-zero pixels
    val_masks = {
        "endrow": [1755, 18334],
        "water": [987, 18334],
        "waterway": [696, 18334],
        "double_plant": [2322, 18334],
        "drydown": [5800, 18334],
        "storm_damage": [89, 18334],
        "nutrient_deficiency": [3883, 18334],
        "planter_skip": [1197, 18334],
        "weed_cluster": [2834, 18334],
    }

    train_masks = {
        "endrow": [4481, 56944],
        "water": [2155, 56944],
        "waterway": [3899, 56944],
        "double_plant": [6234, 56944],
        "drydown": [16806, 56944],
        "storm_damage": [356, 56944],
        "nutrient_deficiency": [13308, 56944],
        "planter_skip": [2599, 56944],
        "weed_cluster": [11111, 56944],
    }

    # train_ratios = {
    #     'endrow': 0.07869134588367518,
    #     'water': 0.037844197808373135,
    #     'waterway': 0.06847077830851363,
    #     'double_plant': 0.10947597639786456,
    #     'drydown': 0.29513205956729416,
    #     'storm_damage': 0.00625175611126721,
    #     'nutrient_deficiency': 0.23370328744029223,
    #     'planter_skip': 0.045641331834785054,
    #     'weed_cluster': 0.19512152289969092
    # }
    # ratio of black to white pixels in the labels

    val_masks = {
        "drydown": [674846434, 4806148096],
        "weed_cluster": [201170131, 4806148096],
        "waterway": [20129377, 4806148096],
        "endrow": [60007223, 4806148096],
        "planter_skip": [10885482, 4806148096],
        "nutrient_deficiency": [343483024, 4806148096],
        "double_plant": [44044947, 4806148096],
        "storm_damage": [4224793, 4806148096],
        "water": [74630112, 4806148096],
    }
    train_masks = {
        "drydown": [1838660590, 14927527936],
        "weed_cluster": [921419728, 14927527936],
        "waterway": [141159490, 14927527936],
        "endrow": [135527899, 14927527936],
        "planter_skip": [28304311, 14927527936],
        "nutrient_deficiency": [1108061290, 14927527936],
        "double_plant": [126330289, 14927527936],
        "storm_damage": [27771992, 14927527936],
        "water": [149637355, 14927527936],
    }
    train_ratios = {
        "drydown": 0.12317247690863742,
        "weed_cluster": 0.061726210257349874,
        "waterway": 0.009456320604805063,
        "endrow": 0.009079058473784792,
        "planter_skip": 0.001896115091618074,
        "nutrient_deficiency": 0.07422938980591301,
        "double_plant": 0.008462907558547275,
        "storm_damage": 0.0018604548669457603,
        "water": 0.010024255566062403,
    }
    # adds up to 1.070332256251756
    # L1 loss with ratios
    # check where the target comes from and see what the ratio is for that
    # class
    drydown_ratio = train_ratios["drydown"]

    mask = torch.max(labels, dim=1, keepdim=True)[0] > 0
    loss_non_zero = torch.mean((1 / drydown_ratio) * torch.abs(pred - labels), dim=1)
    loss_zero = torch.mean((1 / (1 - drydown_ratio)) * torch.abs(pred - labels), dim=1)
    loss = torch.where(mask.squeeze(), loss_non_zero, loss_zero)

    return torch.mean(loss)

def weighted_cross_entropy_loss(pred, labels):
    weights = torch.tensor([0.12317247690863742, 0.061726210257349874,
        0.009456320604805063, 0.009079058473784792, 0.001896115091618074,
        0.07422938980591301, 0.008462907558547275, 0.0018604548669457603,
        0.010024255566062403,
    ])
    pred = pred.unsqueeze(1)
    labels = labels.unsqueeze(0)

    loss = nn.CrossEntropyLoss(weights=weights)
    return loss(pred, labels)

def cross_entropy_loss(pred, labels):
    pred = pred.unsqueeze(1)
    labels = labels.unsqueeze(1)

    loss = nn.CrossEntropyLoss()
    return loss(pred, labels)

# accuracy metric
SMOOTH = 1e-6


def mIoU(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (
        (torch.round(outputs).int() & torch.round(labels).int()).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (
        (torch.round(outputs).int() | torch.round(labels).int()).float().sum((1, 2))
    )  # Will be zzero if both are 0

    iou = (1/9) * (intersection + SMOOTH) / (
        union + SMOOTH
    )  # We smooth our devision to avoid 0/0

    # thresholded = (
    #     torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    # )  # This is equal to comparing with thresolds

    return (
        iou.mean()
    )  # Or thresholded.mean() if you are interested in average across the batch
