from PIL import Image
import torch
import os


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root))))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, self.imgs[idx])
        extension = img_path.split('.')[1]
        img = Image.open(img_path).convert("RGB")

        if extension == 'webp':
            new = os.path.join(self.root, self.imgs[idx].split('.')[0] + ".jpg")
            img.save(new, 'jpeg')
            os.remove(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.imgs)
