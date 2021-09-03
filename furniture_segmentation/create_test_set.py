from home_scenes_dataset import HomeScenesDataset
import torch
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt
from references.detection import utils
from training_utils import get_transform

root = '../dataset_ade20k_filtered'

# use our dataset and defined transformations
dataset = HomeScenesDataset(root, get_transform(train=True))
dataset_test = HomeScenesDataset(root, get_transform(train=False))

# split the dataset in train and test set
train_percentage = 0.8
test_percentage = 1 - train_percentage
train_size = int(train_percentage * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(0)
indices = torch.randperm(len(dataset)).tolist()[-test_size:]
dataset_test = torch.utils.data.Subset(dataset_test, indices)
print(len(indices))
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

i = 0
for img, target in test_loader:
    i += 1
    img = img[0].permute(1,2,0).numpy()
    plt.imsave(f'../test_images/TestImage_{i}.jpg', img)
