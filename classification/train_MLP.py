from classification.classification_utils import Classification_Helper
from classification.MLP_model import HomeScenesClassifier
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def main():
    classification_helper = Classification_Helper()
    try:
        dataset = pd.read_csv('classification/dataset_all_objects.csv')
    except:
        dataset = classification_helper.construct_dataset(all_objects=True)
        np.savetxt("classification/dataset_all_objects.csv", dataset, delimiter=",")

    Y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    dataset = classification_helper.make_balanced(X, Y, dataset)

    Y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=2 / 3, stratify=Y)

    # data loader definition
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    train = np.c_[x_train, y_train]
    test = np.c_[x_test, y_test]
    trainloader = torch.utils.data.DataLoader(train, batch_size=64,
                                              shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test, batch_size=64,
                                             shuffle=False, num_workers=0)

    # training phase
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    net = HomeScenesClassifier(num_objs=len(X.columns), num_classes=11).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 30

    for e in range(epochs):
        for i, x in enumerate(trainloader):
            y = x[:, -1].long()

            x = x[:, :-1]
            x, y = x.to(device), y.to(device)

            y_pred = net(x.float())
            loss = crit(y_pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            corr = 0
        for x in testloader:
            y = x[:, -1].long()
            x = x[:, :-1]
            x, y = x.to(device), y.to(device)
            y_pred = net(x.float())
            corr += (torch.max(y_pred, 1)[1] == y).sum()
            # print(str(corr.item()))
        if e % 10 == 0 or e == epochs - 1:
            print(f"Accuracy for epoch {e}:{corr.item() / test.shape[0]}")

    # ATTENZIONE SE SI CREA CSV CON TUTTI GLI OGGETTI --> NON CARICARE SU GIT!!!


if __name__ == '__main__':
    main()
