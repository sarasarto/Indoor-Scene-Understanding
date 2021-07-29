import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2 as cv
from autoencoder_retrieval.retrieval_dataset import RetrievalDataset
from autoencoder_retrieval.autoencoder import ConvEncoder, ConvDecoder
from autoencoder_retrieval.autoencoder_utils import AutoencoderHelper
import torch.nn as nn
from torch import optim
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# il dataset deve essere una cartella divisa in train e test
# dentro le immagini NON vanno divise per categoria
# CARICATE AUTOENC_DATASET_JPG DAL DRIVE

#root = 'autoenc_data_jpg/'
#train_imgs = os.path.join(root, "train/")

imgs_size = 224
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((imgs_size,imgs_size)),
    transforms.ToTensor()
])

full_dataset = RetrievalDataset('autoenc_data_jpg/train/',  transforms=transform)
train_size = int(0.75 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=5, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=5, shuffle=False
)

full_loader = torch.utils.data.DataLoader(
    full_dataset, batch_size=5, shuffle=False
)

''''
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
'''
ret_helper = AutoencoderHelper(imgs_size)
'''
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

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder, decoder = ret_helper.training_autoencoder(train_loader, device)

# Save the feature representations.
EMBEDDING_SHAPE = (1, 256, 7, 7) # This we know from our encoder

# full dataset invece che train_loader
embedding = ret_helper.create_embedding(encoder, full_loader, EMBEDDING_SHAPE, device)

# Convert embedding to numpy and save them
numpy_embedding = embedding.cpu().detach().numpy()
num_images = numpy_embedding.shape[0]

# Save the embeddings for complete dataset, not just train
flattened_embedding = numpy_embedding.reshape((num_images, -1))
print("dimensioni embedding")
print(flattened_embedding)

#ret_helper.find_similar_images(flattened_embedding, img )

knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(flattened_embedding)


img = cv.imread('sofa.jpg')

cv.imshow('image', img)
cv.waitKey()
img = cv.resize(img[:,: ,:], (224,224), interpolation = cv.INTER_AREA)
print(img.shape)
img = img.reshape(1, 3, 224 , 224)
img = torch.from_numpy(img)
img = img.float()

with torch.no_grad():
    image_embedding = encoder(img.to(device)).cpu().detach().numpy()


embedde_img = image_embedding.reshape((image_embedding.shape[0], -1))
print(embedde_img.shape)

_, indices = knn.kneighbors(embedde_img)
indices_list = indices.tolist()
print(indices_list)

for idx in indices_list[0]:
  print(idx-1)

  img , _ = full_dataset[idx -1]
  #print(img.shape)
  #print(type(img))
  image = img.permute(1, 2, 0)
  plt.imshow(image.numpy())
  plt.show()
