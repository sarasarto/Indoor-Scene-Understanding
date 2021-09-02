from retrieval.method_autoencoder.retrieval_dataset import RetrievalDataset
from retrieval.method_autoencoder.model import EncodeModel
import torch
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

class AutoencHelper():
    def __init__(self) -> None:
        pass

    def retrieval(self, query_image, label):
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])

        query_image = transform(query_image)
        query_image = query_image.squeeze(0).squeeze(0)
        query_image = query_image.permute(1, 2, 0)
        embedded_query = self._create_embedding_single_image(query_image)
        #print(embedded_query.shape)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        embedding = torch.load('retrieval/method_autoencoder/dataset_embedding.pt', map_location=device)
        numpy_embedding = embedding.cpu().detach().numpy()
        num_images = numpy_embedding.shape[0]
        flattened_embedding = numpy_embedding.reshape((num_images, -1))
        #print(embedding.shape)

        knn = NearestNeighbors(n_neighbors=6, metric="cosine")
        knn.fit(flattened_embedding)


        full_dataset = RetrievalDataset('retrieval/grabcut_kaggle_dataset/', transforms=transform)
        #print(full_dataset)
        return self.find_similar(knn, embedded_query, full_dataset)
    
    def find_similar(self , knn,  embedded_img, full_dataset):
        _, indices = knn.kneighbors(embedded_img)
        indices_list = indices.tolist()
        #print(indices_list)
        retrieved_imgs = []

        for (j, idx) in enumerate(indices_list[0]):
            img = full_dataset[idx - 1] #-1 because first element in embedding is random(see create_embedding.py file)
            image = img.permute(1, 2, 0)
            if j != 0:
                #append numpy array
                retrieved_imgs.append(image.numpy())

        return retrieved_imgs

    def _create_embedding_single_image(self, input_image, K=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EncodeModel()
        model_file = 'retrieval/method_autoencoder/trained_model.pt'
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.train(False)

        input_image = input_image.permute(2,0,1).float()
        input_image = input_image.unsqueeze(0)
        input_encode = model(input_image, False)

        image_embedding = input_encode.cpu().detach().numpy()
        embedded_img = image_embedding.reshape((image_embedding.shape[0], -1))
        return embedded_img