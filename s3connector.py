from s3torchconnector import S3MapDataset, S3IterableDataset
from PIL import Image
import torchvision

DATASET_URI = (
    "s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/raw/"
    # "s3://adl-agriculture-vision-2021"
)
REGION = "us-east-1"


def load_image(object):
    img = Image.open(object)
    # img.show()
    return (object.key, torchvision.transforms.functional.pil_to_tensor(img))


iterable_dataset = S3IterableDataset.from_prefix(
    # DATASET_URI + "/train/images", region=REGION, transform=load_image
    DATASET_URI, region=REGION, transform=load_image
)

for key, img in iterable_dataset:
    key, img = next(iter(iterable_dataset))
    print(img.shape)

"""
# S3MapDataset eagerly lists all the objects under the given prefix
# to provide support of random access.
# S3MapDataset builds a list of all objects at the first access to its elements or
# at the first call to get the number of elements, whichever happens first.
# This process might take some time and may give the impression of being unresponsive.
map_dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION)

# Randomly access to an item in map_dataset.
item = map_dataset[0]

# Learn about bucket, key, and content of the object
bucket = item.bucket
key = item.key
content = item.read()
len(content)
"""
