# Modified version of the trainer.py file from 10-423 HW2

from dataset import AgricultureVisionDataset
from torch.utils import data
from torch.optim import Adam
from torchvision.transforms import v2
import torch.nn.functional as F
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
import os


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        model,
        folder,
        *,
        image_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=10000,
        save_every=1000,
        gradient_accumulate_every=2,
        results_folder="./results",
        load_path=None,
        shuffle=True,
        device=None,
        dataset_size=1000,
    ):
        super().__init__()
        self.model = model

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.save_every = save_every

        self.train_folder = os.path.join(folder, "train")
        self.val_folder = os.path.join(folder, "val")

        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float16, scale=True),
                v2.functional.invert,
                v2.Resize(image_size, antialias=None),
            ]
        )
        target_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float16, scale=True),
                lambda x: torch.clamp(x, 0.0, 1.0),
                v2.Resize(image_size, antialias=None),
            ]
        )

        self.ds = AgricultureVisionDataset(
            self.train_folder, transform=transform, target_transform=target_transform
        )
        self.ds = torch.utils.data.Subset(
            self.ds,
            range(0, dataset_size),
        )
        print(f"dataset length: {len(self.ds)}")

        self.dl = data.DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            # num_workers=0,
            drop_last=True,
        )

        self.val_ds = AgricultureVisionDataset(
            self.train_folder,
            transform=transform,  # This is an unusual value!
            target_transform=target_transform,
        )
        self.val_ds = torch.utils.data.Subset(
            self.val_ds, range(dataset_size, 2 * dataset_size)
        )

        self.val_dl = data.DataLoader(
            self.val_ds,
            batch_size=4,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=True,
        )

        self.opt = Adam(model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )
        )

        if load_path != None:
            self.load(load_path)

    def save(self, itrs=None):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f"model.pt"))
        else:
            torch.save(data, str(self.results_folder / f"model_{itrs}.pt"))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])

    def train(self):
        start_step = self.step
        self.model.train()
        self.model.to(self.device)
        """
        Training loop
        
            1. Use wandb.log to log the loss of each step 
                This loss is the average of the loss over accumulation steps 
            2. Save the model every self.save_and_sample_every steps
        """
        milestone = 0
        for self.step in tqdm(range(start_step, self.train_num_steps), desc="steps"):
            u_loss = 0
            for i, (img, labels) in enumerate(self.dl):
                img = img.to(self.device)
                labels = labels.to(self.device)

                pred = self.model(img)

                assert pred.shape == labels.shape
                loss = F.l1_loss(pred, labels)

                if (i + 1) % 10 == 0:
                    wandb.log({"loss_during_epoch": loss.item()})

                u_loss += loss.item()
                (loss).backward()

            # use wandb to log the loss
            wandb.log(
                {"loss_per_epoch": u_loss / len(self.dl.dataset) * self.batch_size}
            )

            self.opt.step()
            self.opt.zero_grad()

            if (self.step + 1) % self.save_every == 0:
                milestone = self.step // self.save_every
                save_folder = str(self.results_folder / f"model_{milestone}")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                self.save()

        print("training completed")

    def test(self):
        num_rounds = 5

        with torch.no_grad():
            for i, (img, labels) in enumerate(self.val_dl):
                print(torch.max(labels))

                if torch.max(labels) < 1e-13:
                    continue

                img = img.to(self.device)
                labels = labels.to(self.device)

                pred = self.model(img)
                loss = F.l1_loss(pred, labels)

                wandb.log(
                    {
                        "val_loss_per_batch": loss,
                        "pred_masks": [wandb.Image(p) for p in pred],
                        "true_masks": [wandb.Image(p) for p in labels],
                        "images": [wandb.Image(i) for i in img],
                    }
                )

                break
