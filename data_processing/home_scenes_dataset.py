import matplotlib
import torch
import os
from PIL import Image
from torch._C import import_ir_module
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt


class HomeScenesDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "training"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "training_masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "training", self.imgs[idx])
        mask_path = os.path.join(self.root, "training_masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask[:,:,2]) #con ADE le istanze sono salvate nel canale B della mask
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]  #il primo valore è "0" e indica le piccole parti nero non segmentate 
                                #che quindi non mi interessano

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask[:,:,2] == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        labels = torch.ones((num_objs,), dtype=torch.int64)    
        for i in range(len(masks)):
            coordinates = np.argwhere(masks[i] == 1)[0]
            R = mask[coordinates[0],coordinates[1],0]
            G = mask[coordinates[0],coordinates[1],1]
            instance_class = (R/10).astype(np.int32)*256+(G.astype(np.int32)) #questo valore rappresenta la classe dell'oggetto
            labels[i] = instance_class

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels # da sistemare i vlaori di ritorno bisogna usare i canali r,g
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


#esempio di funzionamento
dataset = HomeScenesDataset('ADE_20K')
img, target = dataset[0]
boxes = target['boxes']
masks = target['masks']
print('num boxes' + str(len(boxes)))
print('num masks' + str(len(masks)))

import cv2
img = cv2.imread('ADE_20K/training/ADE_train_00000006.jpg')

print(target['labels'])

for i in range(len(boxes)):
    plt.imshow(masks[i])
    plt.show()

    

#cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
#cv2.rectangle(img, (boxes.numpy()[0], boxes.numpy()[1]), (boxes.numpy()[2], boxes.numpy()[3]), (0,255,0), 3)
#cv2.imwrite("my.png",img)