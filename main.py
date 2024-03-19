import torch
from unet import Unet
from trainer import Trainer
import wandb

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    wandb.init(
        project="adl-agriculture-vision-2021",
    )

    model = Unet(
        dim=64,
        dim_mults=[1, 2, 4, 8],
        out_dim=1,
    ).to(device)

    trainer = Trainer(
        model,
        folder="./data/Agriculture-Vision-2021/train/",
        image_size=128,
        train_batch_size=4,
        train_num_steps=1000,
        save_every=1000,
        results_folder="./results/baseline",
        device=device,
    )

    print("beginning training...")
    trainer.train()
