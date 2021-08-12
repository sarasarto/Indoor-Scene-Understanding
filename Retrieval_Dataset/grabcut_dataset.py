import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import json

# script that applies grabcut on all the kaggle images, reading the annotations from the json file
# the classes are  ['lamp', 'sofa', 'armchair', 'chair', 'bed', 'bicycle']
# both the datasets (kaggle_dataset and the new grabcut one) are on Drive

dataset = 'kaggle_dataset_folder_jpg/'
annotations = 'Annotations_Kaggle.json'

data = json.load(open(annotations))
i = 0
for el in data:
    img_name = el['image']

    path_img = dataset + el['annotations'][0]['label'] + '/' + img_name
    coordinates = el['annotations'][0]['coordinates']

    image = cv.imread(path_img)
    x, y, width, height = int(coordinates['x']), int(coordinates['y']), int(coordinates['width']), int(coordinates['height'])

    rect = (x, y, width, height)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    crop_img = image[y:y + height, x:x + width]

    path = 'grabcut_kaggle_dataset/'
    cv.imwrite(os.path.join(path, img_name), crop_img)
    i = i + 1
    if i % 100 == 0:
        print(i)


