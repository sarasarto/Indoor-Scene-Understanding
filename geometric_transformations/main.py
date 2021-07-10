import json
from PIL.Image import new
import cv2
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
import random as rng
from PIL import Image

img_path = 'dataset_ade20k_filtered\images\ADE_train_00011497.jpg'
annotations_path = 'dataset_ade20k_filtered\\annotations\ADE_train_00011497.json'
masks_path = 'dataset_ade20k_filtered\masks\ADE_train_00011497_seg.png'

with open(annotations_path, 'r') as json_file:
    data = json.load(json_file)


segmentation = cv2.imread(masks_path)
instances = np.unique(segmentation[:,:,0])
instances = instances[1:]
print(instances)
masks = segmentation[:,:,0] == instances[:,None,None]

img = cv2.imread(img_path) #(H,W,3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for i,instance in enumerate(instances):
    if instance == 66:
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        print(xmin, xmax, ymin, ymax)

        #new_mask = np.where(masks[i]==0, 0, 255)
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img)
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax[0].add_patch(rect)

        roi = masks[i][ymin:ymax, xmin:xmax]
        img = img[ymin:ymax, xmin:xmax]
        print(img.shape)

        roi = np.where(roi==0, 0, 255)
        roi = np.float32(roi)
        #kernel = np.ones((200, 200))
        #closing = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        img[roi==0] = 255
    


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('RetrievalSIFT/divano.jpg', img)
        #contours, hierarchy = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ax[1].imshow(img, cmap='gray')
        plt.show()






