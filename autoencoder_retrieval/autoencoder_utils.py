import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from autoencoder_retrieval.autoencoder import EncodeModel
from torch.optim.lr_scheduler import StepLR


class AutoencoderHelper():
    def __init__(self, model_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_file = model_file

    def train(self, full_loader):
        model = EncodeModel().to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
        criterion = nn.MSELoss()

        epochs = 50
        for epoch in range(epochs):
            for (i, trainData) in enumerate(full_loader):
                trainData = trainData.to(self.device)
                outputs = model(trainData, True)
                optimizer.zero_grad()
                loss = criterion(outputs, trainData)
                loss.backward()
                optimizer.step()
                torch.save(model.state_dict(), self.model_file)
            print('epoch:{} loss:{:7f}'.format(epoch, loss.item()))
            scheduler.step()
            model.train(False)
            '''for (i, testData) in enumerate(test_loader):
                testData = testData.to(self.device)
                outputs = model(testData, True)
                plt.figure(1)
                testData = testData.to('cpu')
                outputs = outputs.to('cpu')
                break'''
            model.train(True)
        return model

    # Search by picture function
    def create_embedding_single_image(self, inputImage, K=5):
        model = EncodeModel()
        model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        model.train(False)

        inputImage = inputImage.unsqueeze(0)
        inputEncode = model(inputImage, False)

        image_embedding = inputEncode.cpu().detach().numpy()
        embedded_img = image_embedding.reshape((image_embedding.shape[0], -1))
        return embedded_img

    def create_embedding_full_dataset(self, dataset_loader, embedding_dim):
        model = EncodeModel()
        model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        model.train(False)

        embedding = torch.randn(embedding_dim)
        dataset_loader = torch.utils.data.DataLoader(dataset_loader, batch_size=2, shuffle=False)

        with torch.no_grad():
            for (i, testImage) in enumerate(dataset_loader):
                testEncode = model(testImage, False)
                embedding = torch.cat((embedding, testEncode.cpu()), 0)

        return embedding

    def find_similar(self , knn,  embedded_img, full_dataset):
        _, indices = knn.kneighbors(embedded_img)
        indices_list = indices.tolist()
        print(indices_list)

        for (j, idx) in enumerate(indices_list[0]):
            img = full_dataset[idx - 1]
            image = img.permute(1, 2, 0)

            plt.figure(1, figsize=(20, 10))
            plt.subplot(1, 5, j + 1)
            plt.imshow(image.numpy())
            plt.title(idx)

        plt.show()