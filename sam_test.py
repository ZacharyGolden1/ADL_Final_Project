from transformers import pipeline
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np

from data_setup import AgricultureVisionDataset

val_dataset = AgricultureVisionDataset(
    root="./data/Agriculture-Vision-2021/val",
    transform=v2.ToPILImage(),
    target_transform=v2.ToPILImage(),
)

generator = pipeline(model="facebook/sam-vit-base", task="mask-generation")

DATA_PATH = "./data/"

image, true_mask = val_dataset[0]
outputs = generator(image)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


plt.imshow(np.array(image))
ax = plt.gca()
for mask in outputs["masks"]:
    show_mask(mask, ax=ax, random_color=True)
plt.axis("off")
plt.savefig("sam_mask.png")

# save image and true mask as png
image.save("image.png")
true_mask.save("true_mask.png")
