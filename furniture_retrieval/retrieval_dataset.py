from PIL import Image 
import torch
import os

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root))))


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
           img = self.transforms(img)           
        
        return img
        #, target

    def __len__(self):
        return len(self.imgs)