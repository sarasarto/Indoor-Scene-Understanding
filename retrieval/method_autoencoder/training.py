import torch
from torchvision import transforms
from retrieval.method_autoencoder.retrieval_dataset import RetrievalDataset
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from retrieval.method_autoencoder.model import EncodeModel
from torch.optim.lr_scheduler import StepLR
from retrieval.evaluation import RetrievalMeasure


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeModel().to(device)
    model_file = 'trained_model.pt'

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    criterion = nn.MSELoss()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = RetrievalDataset('retrieval/grabcut_kaggle_dataset/', transforms=transform)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=2, shuffle=False)

    epochs = 50
    for epoch in range(epochs):
        for (i, trainData) in enumerate(full_loader):
            trainData = trainData.to(device)
            outputs = model(trainData, True)
            optimizer.zero_grad()
            loss = criterion(outputs, trainData)
            loss.backward()
            optimizer.step()

        
        print('epoch:{} loss:{:7f}'.format(epoch, loss.item()))
        scheduler.step()
   
        torch.save(model.state_dict(), model_file)















    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = RetrievalDataset('grabcut_kaggle_dataset/', transforms=transform)
    data_loader = torch.utils.data.DataLoader(full_dataset, batch_size=2, shuffle=False)

    print("starting training...")
    # model=train(full_loader)
    print("training ended.")

    i = 0
    test_dataset = RetrievalDataset('test_kaggle/', transforms=transform)

    embedding_dim = (1, 256, 7, 7)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # --> this two line to create the embedding file <--
    # do not use these unless you change the dataset

    # embedding = helper.create_embedding_full_dataset(full_dataset, embedding_dim)
    # torch.save(embedding, 'embedding_50_arch2.pt')

    embedding = torch.load('embedding_50_arch2.pt', map_location=device)
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]
    flattened_embedding = numpy_embedding.reshape((num_images, -1))
    print(flattened_embedding.shape)

    knn = NearestNeighbors(n_neighbors=6, metric="cosine")
    knn.fit(flattened_embedding)

    AP_test = []

    for inputImage in test_dataset:
        trueImage = inputImage.squeeze(0).squeeze(0)
        trueImage = trueImage.permute(1, 2, 0)
        plt.imshow(trueImage)
        plt.title('Query Image')
        plt.show()

        embedded_img = helper.create_embedding_single_image(inputImage)
        retrieved_imgs = helper.find_similar(knn, embedded_img, full_dataset)
        single_AP = helper.get_AP_autoencoder(retrieved_imgs, trueImage)

        AP_test.append(single_AP)
        print("AP vector")
        print(AP_test)

        i += 1
        if i > 200:
            break

    #AP_test = [0.7000000000000001, 0.8875, 1.0, 1.0, 1.0, 1.0, 0.8055555555555555, 1.0, 1.0, 1.0, 0.8055555555555555, 0.7000000000000001, 0.5888888888888889, 1.0, 1.0, 1.0, 0.3333333333333333, 1.0]
    print("computing MAP on test kaggle dataset ")
    print(helper.compute_MAP_autoencoder(AP_test))


if __name__ == '__main__':
    main()