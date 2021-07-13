from classification_utils import Classification_Helper
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np

def main():
    classification_helper = Classification_Helper()
    
    try:
        dataset = pd.read_csv('classification/dataset_all_objects.csv')
    except:
        dataset = classification_helper.construct_dataset(all_objects=True)
        np.savetxt("classification/dataset_all_objects.csv", dataset, delimiter=",")

    # ATTENZIONE SE SI CREA CSV CON TUTTI GLI OGGETTI --> NON CARICARE SU GIT!!!
    Y = dataset.iloc[:,-1]
    X = dataset.iloc[:,:-1]

    #rendi il dataset bilanciato
    balanced_dataset = classification_helper.make_balanced(X , Y , dataset)

    # training 
    best_estimator, final_acc, best_params = classification_helper.train_RandomForestClassifier(balanced_dataset)
    print(best_estimator)
    print(final_acc)
    print(best_params)

    #best_estimator.feature_importances_

if __name__ == '__main__':
    main()
