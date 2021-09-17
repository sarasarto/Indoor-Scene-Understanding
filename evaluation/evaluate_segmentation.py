from furniture_segmentation.training_utils import get_transform
from furniture_segmentation.home_scenes_dataset import HomeScenesDataset
import torch
import numpy as np
from references.detection import utils
import utils_eval_segmentation
from furniture_segmentation.prediction_model import PredictionModel

root = '../dataset_ade20k_filtered'

# use our dataset and defined transformations
dataset = HomeScenesDataset(root, get_transform(train=True))
dataset_test = HomeScenesDataset(root, get_transform(train=False))

# split the dataset in train and test set
batch_size_train = 2
batch_size_test = 1
train_percentage = 0.6
test_percentage = 1 - train_percentage
train_size = int(train_percentage * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(0)
indices = torch.randperm(len(dataset)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)


PATH = 'model_mask_default.pt'
is_default = True

num_classes = 1324

APs = []

count = 0
for img in test_loader:
    count += 1
    if count % 10 == 0:
        print(f'Analizing image number: {count}')

    gt_class_id = img[1][0]['labels']

    gt_bbox = img[1][0]['boxes'].detach().numpy()
    gt_mask = img[1][0]['masks'].detach().numpy()
    gt_mask = np.moveaxis(gt_mask, 0, 2)
    tensor_img = img[0][0]

    pm = PredictionModel(PATH, num_classes, is_default)
    results = pm.segment_image(img)

    boxes_pred, masks_pred, labels_pred, scores_pred = pm.extract_furniture(results, 0.0)
    masks_pred = np.moveaxis(masks_pred, 0, 2)

    AP, precisions, recalls, overlaps =\
    utils_eval_segmentation.compute_ap(gt_bbox, gt_class_id, gt_mask,
                     boxes_pred, labels_pred, scores_pred.detach().numpy(), masks_pred)
    APs.append(AP)
    if count % 5 == 0:
        print(f'Actual mean value: {np.mean(APs)}')

print("mAP: ", np.mean(APs))