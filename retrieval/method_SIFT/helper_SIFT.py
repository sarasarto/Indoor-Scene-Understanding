import json
import cv2 as cv
import numpy as np
import os
import pickle
from retrieval.query_expansion_transformations import QueryTransformer

class SIFTHelper():
    def __init__(self, dataset_path='retrieval/kaggle_dataset_folder_jpg', grabcut_path='retrieval/grabcut_kaggle_dataset_folder',
                    annotation_path='retrieval/Annotations_Kaggle.json'):
        self.dataset_path = dataset_path
        self.grabcut_path = grabcut_path
        self.data = json.load(open(annotation_path))

    # function which implements the retrieval
    def retrieval(self, query_image, label):
        qt = QueryTransformer()
        

        obj_list = []
        num_good = []

        # applying geometric transformation to the imag
        transformed = []
        transformed.append(query_image)
        transformed.append(qt.flip_image(query_image, 1))
        transformed.append(qt.rotate_image(query_image, 10))
        transformed.append(qt.rotate_image(query_image, -10))
        transformed.append(qt.scale_img(query_image, 1.5))
        transformed.append(qt.scale_img(query_image, 0.5))

        print("Computing SIFT...")

        des = []
        try:
            #with open('retrieval/method_SIFT/keypoints.pkl', 'rb') as f:
                #kp = pickle.load(f)
            with open('retrieval/method_SIFT/descriptors.pkl', 'rb') as f:
                des = pickle.load(f)
        except FileNotFoundError:
            print('Pickle file not found. Please compute sift descriptors before!')

    
        # compute SIFT for the image and its transformations then compare them with images of the dataset
        for (j, img) in enumerate(transformed):
            gray_l = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray_l, None)

            img_names = []
            for des2, annotated_img in zip(des, self.data):
                #NB:DA RIVEDERE MEGLIO QUESTO IF CON KEVIN
                if annotated_img['annotations'][0]['label'] != label or des2 is None:
                    continue
                
                img_names.append(annotated_img['image'])
                # cv2.BFMatcher() takes the descriptor of one feature in first set
                # and is matched with all other features in second set using some distance calculation.
                # And the closest one is returned.
                bf = cv.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                num_good.append(len(good))
        
        num_good = np.reshape(num_good, [len(transformed), -1]) #matrice con tante righe quante le trasformazioni che facciamo

        num_good = num_good.sum(axis=0)
        num_good = np.array(num_good)

        # Select only the best 3 results
        num_good_sorted = num_good.argsort()
        
        similar_images = []
        for i, img_name in enumerate(img_names):

            if i in num_good_sorted[-5:]:
                path = self.dataset_path + '/' + label + '/'

                path = os.path.join(path, img_name)
                print(path)
                img = cv.imread(path, cv.COLOR_BGR2RGB)
                similar_images.append(img)
        
        return similar_images