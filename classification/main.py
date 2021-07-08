import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import os

def construct_dataset():
    root_path = 'ADE_20K_ufficiale/annotations'
    with open('dataset_processing/dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    dataset_size = dataset_info['dataset_size']
    num_objs = len(dataset_info['instances_per_obj'])
    dataset = np.zeros((dataset_size, num_objs))
    labels = []

    for img, file in enumerate(os.listdir(root_path)):
            with open(os.path.join(root_path, file), 'r') as json_file:
                data = json.load(json_file)
                scene = data['annotation']['scene'][-1]
                labels.append(scene)
                img_objs = data['annotation']['object']

            for obj in img_objs:
                #print(obj)
                #print(data['instances_per_obj'])
                if obj['raw_name'] in dataset_info['instances_per_obj']:
                    idx = list(dataset_info['instances_per_obj']).index(obj['raw_name'])
                    dataset[img, idx] = 1
    #encode labels with sklearn
    lb = LabelEncoder()
    labels = lb.fit_transform(labels) 
    dataset = np.c_[dataset, labels]

    return dataset #include labels in the last column


dataset = construct_dataset()
np.savetxt("classification/dataset.csv", dataset, delimiter=",")
dataset = pd.read_csv('classification/dataset.csv')

Y = dataset.iloc[:,-1]
X = dataset.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=2/3)
unique, counts = np.unique(y_test, return_counts=True)
print(unique, counts)

model = RandomForestClassifier(random_state=0, max_depth=30)

model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(f'Accuracy è: {accuracy_score(y_test, predictions)}')

#TODO: BISOGNA ANCORA GESTIRE LA QUESTIONE SULLO SBILANCIAMENTO DEL DATASET




