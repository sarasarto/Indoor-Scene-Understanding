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

img_path = 'dataset_ade20k_filtered\images\ADE_train_00011497.jpg'
annotations_path = 'dataset_ade20k_filtered\\annotations\ADE_train_00011497.json'
masks_path = 'dataset_ade20k_filtered\masks\ADE_train_00011497_seg.png'


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

'''
keypoints_matrix = np.zeros((len(kp1),2))
for i, keypoint in enumerate(kp1):
    keypoints_matrix[i, 0] = keypoint.pt[0]
    keypoints_matrix[i, 1] = keypoint.pt[1]
'''

test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
rotated = ndimage.rotate(test1, 45)
cv2.imshow('test ruotato', 45)
cv2.waitKey()

#rotation angle in degree


path = 'retrieval_sift/data_sofa/'
num_good = []
for f in os.listdir(path):

    path2 = os.path.join(path, str(f))
    img2 = cv2.imread(path2, cv2.COLOR_BGR2RGB)

    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img2,None)
    test2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2RGB)
    #plt.imshow(test2)
    #plt.show()

    # cv2.BFMatcher() takes the descriptor of one feature in first set 
    # and is matched with all other features in second set using some distance calculation.
    # And the closest one is returned.

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #print("match trovati: " + str(len(matches)))
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    num_good.append(len(good))
    #print("match buoni trovati: " + str(len(good)))
    img3 = cv2.drawMatchesKnn(img,kp1,img2,kp2, good,None, flags=2)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3), plt.show()

# stampo i punti che soddisfano la percentuale per ogni immagine
for i , img in enumerate(os.listdir(path)):
  print(str(img) + '\t' + str(num_good[i]))
num_good = np.array(num_good)

# decido di tenere solo i tre migliori
num_good_sorted = num_good.argsort()

best = []
for i , img in enumerate(os.listdir(path)):
    if i in num_good_sorted[-3:]:
        best.append(str(img))

print("Le migliori tre corrispondenze: " + str(best))

'''
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
#plt.show()
'''