import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
import json
import pandas as pd
import os
import pickle

class Classification_Helper():
    def __init__(self, root_path='ADE_20K/annotations'):
        self.root_path = root_path       
        self.scenes = []
        self.obj_classes = []


    def construct_dataset(self, all_objects=False):
        root_path = self.root_path

        if all_objects:
            file = 'dataset_info_all_objs.json'
        else:
            file = 'dataset_info.json'
        with open('ADE20K_filtering/' + file, 'r') as f:
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
                    if obj['raw_name'] in dataset_info['instances_per_obj']:
                        idx = list(dataset_info['instances_per_obj']).index(obj['raw_name'])
                        dataset[img, idx] = 1
        
        #encode labels with sklearn
        lb = LabelEncoder()
        keys_list = labels
        values_list = lb.fit_transform(labels)
        values_list_str = map(str, values_list)
        zip_iterator = zip(keys_list, values_list_str)
        dictionary = dict(zip_iterator)

        #save mapping on file
        with open('classification/rooms_mapping.json', 'w') as f:
            json.dump(dictionary, f)

        dataset = np.c_[dataset, values_list]

        return dataset #include labels in the last column

    def make_balanced(self, X, Y , dataset):
        x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=2/3)
        unique, counts = np.unique(y_test, return_counts=True)
        data_bal = dataset.copy()
        for c in range(len(unique)):
            # la classe piu grande ha 1600 circa elementi
            # ho deciso di bilanciare solo quelle che avevano metà degli esempi
            if counts[c] <= np.max(counts)/2:
                medium  = resample(dataset[Y==unique[c]], replace=True, n_samples=900)

                data_bal = pd.concat([data_bal, medium])
        return data_bal

    def train_RandomForestClassifier(self , dataset):
        Y = dataset.iloc[:,-1]
        X = dataset.iloc[:,:-1]

        x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=2/3)
        unique, counts = np.unique(y_test, return_counts=True)
        print(unique, counts)

        model = RandomForestClassifier(random_state=0, max_depth=30)

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Default model accuracy: {accuracy}')


        # con il GridSearch troviamo i parametri migliori 
        parameters = {
            'n_estimators' : [10 , 50 , 100],
            'criterion' : ['gini', 'entropy'], 
            'max_depth' : [10 , 30 , 60], 
            'max_features' : ['auto', 'sqrt', 'log2'],
        }

        gs_clf = GridSearchCV(model, parameters)
        gs_clf.fit(x_train, y_train)
        y_pred = gs_clf.predict(x_test)
        accuracy_grid = accuracy_score(y_test, y_pred)
        print(f'Best accuracy after fine-tuning: {accuracy_grid}')
        final_acc = np.max([accuracy, accuracy_grid])
        best_params = gs_clf.best_params_
        best_estimator = gs_clf.best_estimator_
        return best_estimator, final_acc, best_params

    def construct_fv_for_prediction(self, labels):
        with open('ADE20K_filtering/dataset_info_all_objs.json', 'r') as f:
            data = json.load(f)

        with open('ADE20K_filtering/mapping.json',
                  'r') as f:
            mapping = json.load(f)

        num_objs = len(data['instances_per_obj'])
        vector = np.zeros((1,num_objs))
        labels = np.unique(labels) #-1 because labels start from 1 but array indexing from 0
        idxs = []
        for label in labels:
            for map in mapping:
                if mapping[map]['new_label'] == label:
                    print(map)
                    idxs.append(list(data['instances_per_obj']).index(map))
                    print(list(data['instances_per_obj']).index(map))

        vector[:,idxs] = 1
        return vector

    def predict_room(self, vector):
        try:
            with open('classification/randomforest_model.pkl', 'rb') as fid:
                classifier = pickle.load(fid)
        except:

            raise ValueError('Impossibile to load the model. First you must train it!.')
        
        prediction = classifier.predict(vector)
        with open('classification/rooms_mapping.json', 'r') as f:
            room_mapping = json.load(f)
        
        for room in room_mapping:
            code_label = int(room_mapping[room])
            if code_label == prediction:
                return prediction, room
        