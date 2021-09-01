from torchvision import transforms
from retrieval_dataset import RetrievalDataset
import torch
from model import EncodeModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeModel().to(device)
    model_file = 'retrieval/method_autoencoder/trained_model.pt'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = RetrievalDataset('retrieval/grabcut_kaggle_dataset/', transforms=transform)
    embedding_dim = (1, 256, 7, 7)
    

    model.load_state_dict(torch.load(model_file, map_location=device))
    model.train(False)

    embedding = torch.randn(embedding_dim)
    dataset_loader = torch.utils.data.DataLoader(full_dataset, batch_size=2, shuffle=False)

    with torch.no_grad():
        for (i, testImage) in enumerate(dataset_loader):
            testEncode = model(testImage, False)
            embedding = torch.cat((embedding, testEncode.cpu()), 0)

    torch.save(embedding, 'retrieval/method_autoencoder/dataset_embedding.pt')

    
if __name__ == '__main__':
    main()