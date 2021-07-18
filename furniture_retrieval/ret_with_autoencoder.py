import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np

from sklearn.neighbors import NearestNeighbors
from retrieval_dataset import RetrievalDataset
from autoencoder import ConvAutoencoder


# il dataset deve essere una cartella divisa in train e test
# dentro le immagini NON vanno divise per categoria
root = '/content/drive/MyDrive/autoenc_data/'

train_imgs = os.path.join(root, "train/")
test_imgs = os.path.join(root, "test/")

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

dataset_train = RetrievalDataset(train_imgs, transforms=transform )

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=0,
    )

# loading test dataset
dataset_test = RetrievalDataset(test_imgs, transforms=transform )

test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=0,
    )


#Instantiate the model
model = ConvAutoencoder()
print(model)

#Loss function
criterion = nn.MSELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# in questo vettore mi salvo solo i features vector 
# dell 'ultima epoca 
val = []
#Epochs
n_epochs = 40

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    for data in train_loader:
        images= data
        #print(type(images))
        #images = images.unsqueeze(0) 
        #print(images.shape)
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if epoch == n_epochs:
          for i in range(len(data)):
            val.append(outputs[i][0].detach().numpy())
        
        #print(outputs.shape)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        #print(images.size(0))
        train_loss += loss.item()*images.size(0)
          
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

val = np.array(val)
val = val.reshape(len(val),val.shape[1]*val.shape[2])

# questa è una prova di ricostruzione
# ci va questo eval()???
model.eval()


for idx , img in enumerate(test_loader):

  images= img
  plt.imshow(images[0, 0,:,:])
  plt.show()
  images = images.to(device)
  optimizer.zero_grad()
  outputs = model(images)
  outputs = outputs.detach().numpy()
  
  plt.imshow(outputs[0,1,:,:])
  plt.show()
# fine prova



# da qui parte il vero retrieval
# cerchiamo i 5 oggetti simili

knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(val)

# mi prendo una immagine del test_loader per comodita
img = outputs[0]
plt.imshow(img[0, : , :])
img = img[0].reshape(img.shape[1]*img.shape[2],)

_, indices = knn.kneighbors([img] , n_neighbors=5)

for idx in indices[0]:
  print(idx)
  prova = val[idx].reshape(300, 300)

  plt.imshow(prova)
  plt.show()