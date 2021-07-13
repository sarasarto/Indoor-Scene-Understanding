from class_functions import Classification_Helper
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import numpy as np

def main():
    classifier = Classification_Helper()

    #creo il dataset con solo i nostri oggetti
    
    d1 = classifier.construct_dataset()
    np.savetxt("FEW_OBJS_dataset.csv", d1, delimiter=",")

    #creo il dataset con tutti gli oggetti delle scene che ci interessano
    
    #d_all = classifier.construct_dataset_ALL_OBJS()
    #np.savetxt("ALL_OBJS_dataset.csv", d_all, delimiter=",")


    #leggo uno dei csv che ho
    # --> modifica il nome csv
    # ATTENZIONE SE SI CREA CSV CON TUTTI GLI OGGETTI --> NON CARICARE SU GIT!!!
    dataset = pd.read_csv('classification/dataset.csv')
    Y = dataset.iloc[:,-1]
    X = dataset.iloc[:,:-1]

    #rendi il dataset bilanciato
    balanced_dataset = classifier.make_balanced(X , Y , dataset)

    # training 
    accuracy, model = classifier.train_RandomForestClassifier(balanced_dataset)
    print(str(accuracy))

    model.feature_importances_


if __name__ == '__main__':
    main()
