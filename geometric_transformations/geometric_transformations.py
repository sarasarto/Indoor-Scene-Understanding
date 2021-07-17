import json
from PIL.Image import new
import cv2
import os
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
import random as rng
from PIL import Image
from scipy import ndimage
from numpy.lib.type_check import _imag_dispatcher


class GeometryTransformer():
    def __init__(self):
        pass

    def transform_bbox(self, bbox):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img,None)



        keypoints_matrix = np.zeros((len(kp1),2))
        for i, keypoint in enumerate(kp1):
            keypoints_matrix[i, 0] = keypoint.pt[0]
            keypoints_matrix[i, 1] = keypoint.pt[1]


        test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(test1)
        plt.show()


        tl = (np.min(keypoints_matrix[:,0]), np.max(keypoints_matrix[:,1]))
        tr = (np.max(keypoints_matrix[:,0]), np.max(keypoints_matrix[:,1]))
        br = (np.max(keypoints_matrix[:,0]), np.min(keypoints_matrix[:,1]))
        bl = (np.min(keypoints_matrix[:,0]), np.min(keypoints_matrix[:,1]))

        x = [tl[0], tr[0], br[0], bl[0]]
        #print(x)
        y = [tl[1], tr[1], br[1], bl[1]]
        print(y)

        #test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #print(kp1[0].pt)
        implot = plt.imshow(img)
        plt.scatter(x,y, c='r')
        plt.title("keypoints del letto")
        plt.show()


'''
segmentation = cv2.imread(masks_path)
instances = np.unique(segmentation[:,:,0])
instances = instances[1:]
print(instances)
masks = segmentation[:,:,0] == instances[:,None,None]

#considero per il momento solo l'istanza del letto
#che ha indice 11
pos = np.where(masks[8])
xmin = np.min(pos[1])
xmax = np.max(pos[1])
ymin = np.min(pos[0])
ymax = np.max(pos[0])

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
roi = masks[8][ymin:ymax, xmin:xmax]
img = img[ymin:ymax, xmin:xmax]
roi = np.where(roi==0, 0, 255)
roi = np.float32(roi)
img[roi==0] = 255


rotated = ndimage.rotate(img, 45)
print(rotated)
cv2.imshow('immagine ruotata', rotated)
cv2.waitKey()


sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img,None)



keypoints_matrix = np.zeros((len(kp1),2))
for i, keypoint in enumerate(kp1):
    keypoints_matrix[i, 0] = keypoint.pt[0]
    keypoints_matrix[i, 1] = keypoint.pt[1]


test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(test1)
plt.show()


tl = (np.min(keypoints_matrix[:,0]), np.max(keypoints_matrix[:,1]))
tr = (np.max(keypoints_matrix[:,0]), np.max(keypoints_matrix[:,1]))
br = (np.max(keypoints_matrix[:,0]), np.min(keypoints_matrix[:,1]))
bl = (np.min(keypoints_matrix[:,0]), np.min(keypoints_matrix[:,1]))

x = [tl[0], tr[0], br[0], bl[0]]
#print(x)
y = [tl[1], tr[1], br[1], bl[1]]
print(y)

#test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#print(kp1[0].pt)
implot = plt.imshow(img)
plt.scatter(x,y, c='r')
plt.title("keypoints del letto")
plt.show()
'''