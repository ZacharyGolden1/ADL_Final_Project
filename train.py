"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch
import data_setup, engine, utils, loss
from unet import Unet
import argparse
import wandb

from torchvision.transforms import v2


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
        "--train_dir", default="./data/Agriculture-Vision-2021/train", type=str
    )
    parser.add_argument(
        "--test_dir", default="./data/Agriculture-Vision-2021/val", type=str
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
    parser.add_argument("--dataset_size", default=None, type=int)
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="The number of iterations for training.",
    )
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--save_every", default=25, type=int)

    args = parser.parse_args()

    print("Arguments")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("device:", device)
    print()

    img_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.functional.invert,
            v2.Resize(args.image_size, antialias=None),
        ]
    )
    mask_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            lambda x: torch.clamp(x, 0.0, 1.0),
            v2.Resize(args.image_size, antialias=None),
        ]
    )

    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=img_transform,
        target_transform=mask_transform,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
    )

    model = Unet(
        dim=64,
        dim_mults=[1, 2, 4, 8],
        out_dim=1,
    ).to(device)

    if args.load_path:
        print("Loading model from", args.load_path)
        model.load_state_dict(torch.load(args.load_path))

    # Set loss and optimizer
    # loss_fn = torch.nn.L1Loss()
    loss_fn = loss.weighted_l1_loss

    acc_fn = loss.mIoU

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize wandb
    wandb.init(
        entity="slumba-cmu",
        project="agriculture-vision-adl",
        name=args.save_dir,
        config={
            "model": model.__class__.__name__,
            "loss": loss_fn.__class__.__name__,
            **vars(args),
        },
    )

    # Start training with help from engine.py
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_every=args.save_every,
        save_dir=args.save_dir,
    )

    # Save the model with help from utils.py
    utils.save_model(
        model=model,
        target_dir=args.save_dir,
        model_name="model.pt",
    )


main()
