from furniture_segmentation.training_utils import get_transform
from furniture_segmentation.home_scenes_dataset import HomeScenesDataset
import torch
from references.detection import utils

root = '../dataset_ade20k_filtered'

# use our dataset and defined transformations
dataset = HomeScenesDataset(root, get_transform(train=True))
dataset_test = HomeScenesDataset(root, get_transform(train=False))

# split the dataset in train and test set
batch_size_train = 2
batch_size_test = 2
train_percentage = 0.8
test_percentage = 1 - train_percentage
train_size = int(train_percentage * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(0)
indices = torch.randperm(len(dataset)).tolist()
#dataset = torch.utils.data.Subset(dataset, indices[0:train_size])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

