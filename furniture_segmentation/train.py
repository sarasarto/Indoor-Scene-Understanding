from vision.references.detection.engine import train_one_epoch
import vision.references.detection.transforms as T
from home_scenes_dataset import HomeScenesDataset
import torchvision
#from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def collate_fn(batch):
    return tuple(zip(*batch))
    

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


dataset = HomeScenesDataset('dataset_ade20k_filtered', get_transform(train=True))
dataset_test = HomeScenesDataset('dataset_ade20k_filtered', get_transform(train=False))

# split the dataset in train and test set
batch_size_train = 2
batch_size_test = 2
train_percentage = 1
test_percentage = 1 - train_percentage
train_size = int(train_percentage * len(dataset))
test_size = len(dataset) - train_size

indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[0:train_size])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

# define training and validation data loaders
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size_train, shuffle=True, num_workers=0,
    collate_fn=collate_fn)

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0,
    collate_fn=collate_fn)


# get the model using our helper function
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 102
model = get_instance_segmentation_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# let's train it for 10 epochs
num_epochs = 1

for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)