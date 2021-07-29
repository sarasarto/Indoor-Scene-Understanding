import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from autoencoder_retrieval.autoencoder import ConvEncoder, ConvDecoder


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

    def training_autoencoder(self, train_loader, device):

        print('starting training')
        # in questo vettore mi salvo solo i features vector
        # dell 'ultima epoca

        device = device
        encoder = ConvEncoder()  # Our encoder model
        decoder = ConvDecoder()  # Our decoder model
        # Shift models to GPU
        encoder.to(device)
        decoder.to(device)
        autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(autoencoder_params, lr=1e-3)  # Adam Optimizer
        loss_fn = nn.MSELoss()

        '''# Loss function
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
        return val, model'''
        n_epochs = 1
        #  Set networks to train mode.
        for epoch in range(1, n_epochs + 1):
            # monitor training loss
            train_loss = self.train_step(
                encoder, decoder, train_loader, loss_fn, optimizer, device=device
            )

            # Training
            print(f"Epochs = {epoch}, Training Loss : {train_loss}")

        return encoder, decoder

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

    def create_embedding(self, encoder, full_loader, embedding_dim, device):
        """
        Creates embedding using encoder from dataloader.
        encoder: A convolutional Encoder. E.g. torch_model ConvEncoder
        full_loader: PyTorch dataloader, containing (images, images) over entire dataset.
        embedding_dim: Tuple (c, h, w) Dimension of embedding = output of encoder dimesntions.
        device: "cuda" or "cpu"
        Returns: Embedding of size (num_images_in_loader + 1, c, h, w)
        """
        # Set encoder to eval mode.
        encoder.eval()
        # Just a place holder for our 0th image embedding.
        embedding = torch.randn(embedding_dim)

        # Again we do not compute loss here so. No gradients.
        with torch.no_grad():
            for batch_idx, (train_img, target_img) in enumerate(full_loader):
                # We can compute this on GPU. be faster
                train_img = train_img.to(device)

                # Get encoder outputs and move outputs to cpu
                enc_output = encoder(train_img).cpu()
                # print(enc_output.shape)
                # Keep adding these outputs to embeddings.
                embedding = torch.cat((embedding, enc_output), 0)

        # Return the embeddings
        return embedding

    def train_step(self, encoder, decoder, train_loader, loss_fn, optimizer, device):
        # device = "cuda"
        encoder.train()
        decoder.train()

        # print(device)

        for batch_idx, (train_img, target_img) in enumerate(train_loader):
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            optimizer.zero_grad()

            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            loss = loss_fn(dec_output, target_img)
            loss.backward()

            optimizer.step()

        return loss.item()
