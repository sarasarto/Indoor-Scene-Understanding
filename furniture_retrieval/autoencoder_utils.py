import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch.nn as nn


class AutoencoderHelper():
    def __init__(self, imgs_size):
        self.imgs_size = imgs_size

    def find_similar_images(self, vector, img, similar=5, metric='cosine'):
        # trying to find the 5 nearest object
        # we fit on val that is the feature vectors of the autoencoder
        print("finding similar objects")
        knn = NearestNeighbors(n_neighbors=similar, metric=metric)
        knn.fit(vector)

        _, indices = knn.kneighbors([img], n_neighbors=similar)

        return indices

    def show_similar(self, vectors, indices):

        for idx in indices[0]:
            print(idx)
            prova = vectors[idx].reshape(self.imgs_size, self.imgs_size)

            plt.imshow(prova)
            plt.show()

    def training_autoencoder(self, model, train_loader):

        print('starting training')
        # in questo vettore mi salvo solo i features vector
        # dell 'ultima epoca

        # Loss function
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        val = []
        n_epochs = 1

        for epoch in range(1, n_epochs + 1):
            # monitor training loss
            train_loss = 0.0

            # Training
            for data in train_loader:
                images = data
                images = images.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                # saving only the last reconstructed results
                if epoch == n_epochs:
                    for i in range(len(data)):
                        val.append(outputs[i][0].detach().numpy())

                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

        val = np.array(val)
        val = val.reshape(len(val), val.shape[1] * val.shape[2])
        return val, model

    def get_class_vector(self, model, dataset_classe):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("getting class vector..")
        val_classe = []  # salvo i feauture vector solo della classe
        for idx, img in enumerate(dataset_classe):
            images = img
            images = torch.tensor(images).unsqueeze(0)

            outputs = model(images)
            outputs = outputs.detach().numpy()

            val_classe.append(outputs[0][0])
        val_classe = np.array(val_classe)
        val_classe = val_classe.reshape(len(val_classe), val_classe.shape[1] * val_classe.shape[2])
        return val_classe
