import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import json
from retrieval.query_expansion_transformations import QueryTransformer
from retrieval.evaluation import RetrievalMeasure
from PIL import Image


class SIFT_Helper():
    def __init__(self, dataset_path='kaggle_dataset_folder_jpg', grabcut_path='grabcut_kaggle_dataset_folder',
                 ann_path='Annotations_Kaggle.json'):
        self.dataset_path = dataset_path
        self.grabcut_path = grabcut_path
        self.ann_path = ann_path
        self.rm = RetrievalMeasure()

    # preparazione immagine sui cui fare retrieval
    # si ritorna l'immagine con grabcut
    def load_image(self, box, img):
        image = cv.imread(img)
        x, y, width, height = int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])
        rect = (x, y, width, height)
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]
        crop_img = image[y:y + height, x:x + width]
        plt.imshow(crop_img), plt.colorbar(), plt.show()

        return crop_img

    # function which implements the retrieval
    def retrieval(self, image, label):

        gs = QueryTransformer()

        obj_list = []
        num_good = []

        # applying geometric transformation to the image
        transformed = []
        transformed.append(image)
        transformed.append(gs.flip_image(image, 1))
        transformed.append(gs.rotate_image(image, 10))
        transformed.append(gs.rotate_image(image, -10))
        transformed.append(gs.scale_img(image, 1.5))
        transformed.append(gs.scale_img(image, 0.5))

        print("Calcolando i SIFT...")

        data = json.load(open(self.ann_path))

        des = []
        kp = []

        # compute SIFT for images of the correct class of Retrieval Dataset
        for im in data:

            if im["annotations"][0]["label"] == label:

                # ho aggiunto questo path
                path = self.grabcut_path + '/' + label
                path2 = os.path.join(path, im["image"])

                if os.path.isfile(path2):
                    obj_list.append(im["image"])

                    img2 = cv.imread(path2, cv.COLOR_BGR2RGB)
                    gray_2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

                    sift = cv.SIFT_create()

                    kp2, des2 = sift.detectAndCompute(gray_2, None)
                    kp.append(kp2)
                    des.append(des2)

                    # code to draw keypoints on the image
                    # test2 = cv.drawKeypoints(img2, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    # plt.imshow(test2)
                    # plt.show()

        # compute SIFT for the image and its transformations then compare them with images of the dataset
        for (j, img) in enumerate(transformed):

            gray_l = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray_l, None)

            test = cv.drawKeypoints(img, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # plt.imshow(test)
            # plt.title("keypoints arredo immagine principale")
            # plt.show()

            plt.figure(1, figsize=(20, 10))
            plt.subplot(1, len(transformed), j + 1)
            plt.imshow(test)
            plt.suptitle("Transformed Images and their key points")

            for kp2, des2, img2 in zip(kp, des, obj_list):
                # cv2.BFMatcher() takes the descriptor of one feature in first set
                # and is matched with all other features in second set using some distance calculation.
                # And the closest one is returned.

                if des2 is None:
                    continue
                bf = cv.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                num_good.append(len(good))

                # code to view the matches between images SIFT
                # path = os.path.join(self.grabcut_path, img2)
                # img2 = cv.imread(path, cv.COLOR_BGR2RGB)
                # img3 = cv.drawMatchesKnn(image, kp1, img2, kp2, good, None, flags=2)
                # plt.imshow(img3), plt.show()

        plt.show()
        print(len(obj_list))
        print(len(num_good))
        return obj_list, num_good

    # print retrieval results
    def print_results(self, obj_list, num_good, label):
        # sum (for each image) of scores obtained with different geometric transformations
        num_good = np.reshape(num_good, [6, -1])
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

    def find_similar_and_print(self, test_image):

        for img in os.listdir(test_image):
            path = test_image + '/' + img
            plt.imshow(Image.open(path))
            plt.title('Query Image')
            plt.show()

            label = input('Write the class label [lamp, sofa, armchair, chair, bed, bicycle]: ')
            my_image = cv.imread(path)
            # non faccio controlli che sia scritto bene, è solo una prova
            # avrei potuto cercare immagine nell annotation di kaggle ma tanto poi useremo la mask

            obj_list, num_good = self.retrieval(my_image, label)
            self.print_results(obj_list, num_good, label)

    def ask_evaluation(self, test_image):
        eval = input('Do you want to evaluate the method?? [y/n]: ')
        if eval == 'y':
            self.do_evaluation(test_image)

    def do_evaluation(self, test_image):
        AP_test = []
        for img in os.listdir(test_image):
            path = test_image + '/' + img
            plt.imshow(Image.open(path))
            plt.title('Query Image')
            plt.show()

            label = input('Write the class label [lamp, sofa, armchair, chair, bed, bicycle]: ')
            my_image = cv.imread(path)
            # non faccio controlli che sia scritto bene, è solo una prova
            # avrei potuto cercare immagine nell annotation di kaggle ma tanto poi useremo la mask

            obj_list, num_good = self.retrieval(my_image, label)

            retr_imgs = self.print_results(obj_list, num_good, label)

            # starting evaluation
            single_AP = self.get_AP_SIFT(retr_imgs, img)
            AP_test.append(single_AP)
            print("AP vector")
            print(AP_test)

        print("computing MAP on test kaggle dataset ")
        print(self.compute_MAP_SIFT(AP_test))

    def get_AP_SIFT(self, retrieved_imgs, img_ref):

        user_resp = self.rm.get_user_relevance_retrieval(retrieved_imgs)
        print("user responses:")
        print(user_resp)
        AP = self.rm.get_AP(user_resp, 5)
        print("Average Precision: " + str(AP))

        return AP

    def compute_MAP_SIFT(self, AP_vector):
        return self.rm.compute_MAP(AP_vector)

# esempio
# rs = SIFT_Helper()
# bounding_box = [287, 252, 583, 440]
# my_image = rs.load_image(bounding_box, 'ADE_train_00000278.jpg')
# obj_list, num_good = rs.retrieval(my_image, "bed")
# rs.print_results(obj_list, num_good, "bed")
