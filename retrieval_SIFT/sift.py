import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import json
from geometric_transformations import GeometryTransformer



class retrieval_sift():
    def __init__(self, dataset_path = '/Users/kevinmarchesini/Documents/RetrievalDataset/downloads/', grabcut_path = 'retrieval_grabcut', ann_path = 'Annotations.json'):
        self.dataset_path = dataset_path
        self.grabcut_path = grabcut_path
        self.ann_path = ann_path

    #preparazione immagine sui cui fare retrieval
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

    #function which implements the retrieval
    def retrieval(self, image, label):

        gs = GeometryTransformer()

        obj_list = []
        num_good = []

        #applying geometric transformation to the image
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

        #compute SIFT for images of the correct class of Retrieval Dataset
        for im in data:
            if im["annotations"][0]["label"] == label:
                path2 = os.path.join(self.grabcut_path, im["image"])
                if os.path.isfile(path2):
                    obj_list.append(im["image"])
                    #print(im["image"])
                    img2 = cv.imread(path2, cv.COLOR_BGR2RGB)
                    gray_2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

                    sift = cv.SIFT_create()
                    kp2, des2 = sift.detectAndCompute(gray_2, None)
                    kp.append(kp2)
                    des.append(des2)

                    # code to draw keypoints on the image
                    #test2 = cv.drawKeypoints(img2, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    #plt.imshow(test2)
                    #plt.show()

        #compute SIFT for the image and its transformations then compare them with images of the dataset
        for img in transformed:
            gray_l = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray_l, None)
            test = cv.drawKeypoints(img, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.imshow(test)
            plt.title("keypoints arredo immagine principale")
            plt.show()


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

                #code to view the matches between images SIFT
                #path = os.path.join(self.grabcut_path, img2)
                #img2 = cv.imread(path, cv.COLOR_BGR2RGB)
                #img3 = cv.drawMatchesKnn(image, kp1, img2, kp2, good, None, flags=2)
                #plt.imshow(img3), plt.show()

        return obj_list, num_good

    #print retrieval results
    def print_results(self, obj_list, num_good, label):
        # sum (for each image) of scores obtained with different geometric transformations
        num_good = np.reshape(num_good, [6, -1])
        num_good = num_good.sum(axis=0)

        #print scores of images
        for i, img in enumerate(obj_list):
            print(str(img) + '\t' + str(num_good[i]))
        num_good = np.array(num_good)

        # Select only the best 3 results
        num_good_sorted = num_good.argsort()

        best = []
        for i, img in enumerate(obj_list):
            if i in num_good_sorted[-3:]:
                path = self.dataset_path + label + '/'
                best.append(str(img))
                path = os.path.join(path, img)
                img = cv.imread(path, cv.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.show()

        print("Le migliori tre corrispondenze: " + str(best))


#esempio

rs = retrieval_sift()
bounding_box = [287, 252, 583, 440]
my_image = rs.load_image(bounding_box, 'ADE_train_00000278.jpg')
obj_list, num_good = rs.retrieval(my_image, "bed")
rs.print_results(obj_list, num_good, "bed")

