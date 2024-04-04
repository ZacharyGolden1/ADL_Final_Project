"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, utils
import argparse

from torchvision import transforms


def seed_torch(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        default=50000,
        type=int,
        help="The number of iterations for training.",
    )
    parser.add_argument(
        "--train_dir",
        default="./Agriculture-Vision-2021/train",
        type=str,
    )
    parser.add_argument(
        "--test_dir",
        default="./Agriculture-Vision-2021/test",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="./results",
        type=str,
        help="The directory to save the trained model.",
    )
    parser.add_argument(
        "--load_path",
        default=None,
        type=str,
        help="The path to load a pre-trained model.",
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backend.mps.is_available() else "cpu"
    )

    img_transform = ...

    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=img_transform,
        target_transform=...,
        batch_size=args.batch_size,
    )

    model = ...
    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Start training with help from engine.py
    # TODO: periodically save
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
    )

    # Save the model with help from utils.py
    utils.save_model(
        model=model,
        target_dir=args.save_dir,
        model_name="model.pt",
    )
