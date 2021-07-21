import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2 as cv
from furniture_retrieval.retrieval_dataset import RetrievalDataset
from furniture_retrieval.autoencoder import ConvAutoencoder
from furniture_retrieval.autoencoder_utils import AutoencoderHelper

# il dataset deve essere una cartella divisa in train e test
# dentro le immagini NON vanno divise per categoria
# CARICATE AUTOENC_DATASET_JPG DAL DRIVE

root = 'autoenc_data_jpg/'
train_imgs = os.path.join(root, "train/")

imgs_size = 300
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((imgs_size,imgs_size)),
    transforms.ToTensor()
])

# loading train dataset
dataset_train = RetrievalDataset(train_imgs, transforms=transform)
train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=0,
    )

# loading dataset sedie
dataset_sedie = RetrievalDataset('dataset_sedie_prova',  transforms=transform)
sedie_loader = torch.utils.data.DataLoader(
    dataset_sedie, batch_size=2, shuffle=False, num_workers=0,
    )

ret_helper = AutoencoderHelper(imgs_size)

#Instantiate the model
model = ConvAutoencoder()
print(model)

#TO DO salvare modello
# training the model on the entire dataset --> do not change!
val, model = ret_helper.training_autoencoder(model, train_loader)

print("val shape")
print(val.shape)


# simulation of the retrieval pipeline
# loading the image
img = cv.imread('sedia.jpg')
print(img.shape)
cv.imshow('starting image', img)
cv.waitKey()
resized = cv.resize(img[:, : , 0], (imgs_size,imgs_size), interpolation = cv.INTER_AREA)
img = resized.reshape(imgs_size*imgs_size,)
print(img.shape)


# if i want to compare with ALL the dataset --> do nothing. you already have vet got after training
# if i want to compare only with a single object class:
# 1. --> dataset_sedie or dataset_class
# 2. --> uncomment lines below


# in order to get the feature vector of the single class
#val = ret_helper.get_class_vector(model, dataset_sedie)
#print("val_classe shape")
#print(val.shape)



# getting the idx of similar objs
idx_similar = ret_helper.find_similar_images(val , img)
print("returned similar idx:")
print(idx_similar)
ret_helper.show_similar(val, idx_similar)



