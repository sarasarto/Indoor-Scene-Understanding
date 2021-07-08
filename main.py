from data_processing.home_scenes_dataset import HomeScenesDataset, get_transform
from references.detection import utils
from references.detection.engine import train_one_epoch, evaluate
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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


dataset = HomeScenesDataset('ADE_20K', get_transform(train=True))
dataset_test = HomeScenesDataset('ADE_20K', get_transform(train=False))

'''
img, target = dataset[0]
#print(img)
#print(target)
img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
img.show()
'''


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

'''
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 53

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)


img, _ = dataset_test[0]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
    print(prediction)
    print(prediction['masks'])
    print(prediction['masks'].shape)


im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
im.show()

im = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
im.show()


print('ook')
'''


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
#print(output)
model.eval()
predictions = model(images)
#print(predictions)


# For inference
model.eval()
x = dataset[1]
predictions = model([x[0]])           # Returns predictions
print(predictions)

import cv2
img = cv2.imread('ADE_20K/training/ADE_train_00000007.jpg')

for i in range((predictions[0]['boxes']).shape[0]):
    rectangle = predictions[0]['boxes'][i]
    image = cv2.rectangle(img, (rectangle[0],rectangle[3]), (rectangle[1],rectangle[2]), (255, 0, 0), 2)
    cv2.imshow('Image', image)