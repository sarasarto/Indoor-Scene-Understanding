import torchvision_mine
from torchvision_mine.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision_mine.models.detection.mask_rcnn import MaskRCNNPredictor
from references.detection import transforms as T

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision_mine.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
  
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #print(f'in_features: {in_features}')
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    #in_features_mask = model.roi_heads.mask_predictor.conv5_mask1.in_channels
    #print(f'in_features_mask: {in_features_mask}')
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256,
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