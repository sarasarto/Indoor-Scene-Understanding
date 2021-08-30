import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from retrieval.query_expansion_transformations import QueryTransformer
from retrieval.evaluation import RetrievalMeasure
from PIL import Image


class SIFTHelper():
    def __init__(self, dataset_path='retrieval/kaggle_dataset_folder_jpg', grabcut_path='retrieval/grabcut_kaggle_dataset_folder',
                 ann_path='retrieval/Annotations_Kaggle.json'):
        self.dataset_path = dataset_path
        self.grabcut_path = grabcut_path
        self.ann_path = ann_path
        self.rm = RetrievalMeasure()

    # function which implements the retrieval
    def retrieval(self, query_image, label):
        qt = QueryTransformer()

        obj_list = []
        num_good = []

        # applying geometric transformation to the image
        transformed = []
        transformed.append(query_image)
        transformed.append(qt.flip_image(query_image, 1))
        transformed.append(qt.rotate_image(query_image, 10))
        transformed.append(qt.rotate_image(query_image, -10))
        transformed.append(qt.scale_img(query_image, 1.5))
        transformed.append(qt.scale_img(query_image, 0.5))

        print("Computing SIFT...")

        des = []
        kp = []
        try:
            with open('retrieval/method_SIFT/keypoints.pkl', 'rb') as f:
                kp = pickle.load(f)
            with open('retrieval/method_SIFT/descriptors.pkl', 'rb') as f:
                des = pickle.load(f)
        except FileNotFoundError:
            print('Pickle file not found. Please compute sift descriptors before!')

    
        # compute SIFT for the image and its transformations then compare them with images of the dataset
        for (j, img) in enumerate(transformed):
            gray_l = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray_l, None)
            test = cv.drawKeypoints(img, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
            plt.figure(1, figsize=(20, 10))
            plt.subplot(1, len(transformed), j + 1)
            plt.imshow(test)
            plt.suptitle("Transformed Images and their key points")

            for kp2, des2, img2 in zip(kp, des, obj_list):
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
        plt.show()

        return obj_list, num_good

    # print retrieval results
    def print_results(self, obj_list, num_good, label):
        # sum (for each image) of scores obtained with different geometric transformations
        num_good = np.reshape(num_good, [5, -1]) #5 perche abbiamo scelto di tornare i primi 5 risultati
        num_good = num_good.sum(axis=0)

        # print scores of images
        # for i, img in enumerate(obj_list):
        #    print(str(img) + '\t' + str(num_good[i]))
        num_good = np.array(num_good)

        # Select only the best 3 results
        num_good_sorted = num_good.argsort()

        best = []
        retr = []

        j = 0
        for i, img in enumerate(obj_list):

            if i in num_good_sorted[-6:]:
                path = self.dataset_path + '/' + label + '/'

                best.append(str(img))
                path = os.path.join(path, img)
                img = cv.imread(path, cv.COLOR_BGR2RGB)
                retr.append(img)

                plt.figure(1, figsize=(20, 10))
                plt.subplot(1, 6, j + 1)
                plt.imshow(img)
                plt.suptitle("5 most similar images")
                j = j + 1
        plt.show()

        print("Le migliori cinque corrispondenze: " + str(best))
        return retr