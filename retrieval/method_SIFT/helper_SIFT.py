import json
import cv2 as cv
import numpy as np
import os
import pickle



class SIFTHelper():
    def __init__(self, dataset_path='retrieval/dataset_retrieval_folder',
                 grabcut_path='retrieval/grabcut_dataset_folder',
                 annotation_path='retrieval/annotations.json'):
        self.dataset_path = dataset_path
        self.grabcut_path = grabcut_path
        self.data = json.load(open(annotation_path))

    # function which implements the retrieval
    def retrieval(self, query_image, label):

        num_good = []
        print("Computing SIFT...")

        des = []
        try:

            with open('retrieval/method_SIFT/descriptors.pkl', 'rb') as f:
                des = pickle.load(f)
        except FileNotFoundError:
            print('Pickle file not found. Please compute sift descriptors before!')

        # compute SIFT for the image then compare them with images of the dataset

        gray_l = cv.cvtColor(query_image, cv.COLOR_RGB2GRAY)
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_l, None)

        img_names = []
        for des2, annotated_img in zip(des, self.data):
            # only images with the same label class are taken
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

        num_good = np.array(num_good)

        # Select only the best 5 results
        num_good_sorted = num_good.argsort()

        similar_images = []
        for i, img_name in enumerate(img_names):

            if i in num_good_sorted[-5:]:
                path = self.dataset_path + '/' + label + '/'

                path = os.path.join(path, img_name)
                img = cv.imread(path, cv.COLOR_BGR2RGB)
                similar_images.append(img)

        return similar_images