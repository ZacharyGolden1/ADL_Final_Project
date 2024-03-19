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

    BATCH_SIZE = 8
    LR = 2e-5
    IMAGE_SIZE = 128
    EPOCHS = 150
    DATASET_SIZE = 1024
    TEST = False

    wandb.init(
        entity="slumba-cmu",
        project="agriculture-vision-adl",
        config={
            "batch size": BATCH_SIZE,
            "learning rate": LR,
            "image size": IMAGE_SIZE,
            "epochs": EPOCHS,
            "dataset size": DATASET_SIZE,
            "testing": TEST,
        },
    )

    model = Unet(
        dim=64,
        dim_mults=[1, 2, 4, 8],
        out_dim=1,
    ).to(device)

    trainer = Trainer(
        model,
        folder="./data/Agriculture-Vision-2021",
        image_size=IMAGE_SIZE,
        train_batch_size=BATCH_SIZE,
        train_lr=LR,
        train_num_steps=EPOCHS,
        save_every=25,
        results_folder="./results/baseline",
        device=device,
        dataset_size=DATASET_SIZE,
        # Add this while testing
        load_path="./results/baseline/model.pt" if TEST else None,
    )

    if not TEST:
        print(f"beginning training on {device}...")
        trainer.train()
    else:
        print("testing...")
        trainer.test()
