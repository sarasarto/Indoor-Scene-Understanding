import cv2 as cv
import os
import json

# grabcut for each image in the folder
# the classes are  ['lamp', 'sofa', 'armchair', 'chair', 'bed', 'bicycle']

annotations = 'Annotations_Kaggle.json'
dataset = 'grabcut_kaggle_dataset/'     # dataset with 1809 images
data = json.load(open(annotations))

for el in os.listdir(dataset):

    img_name = el
    for ann in data:
        if img_name == ann['image']:
            label = ann['annotations'][0]['label']

            path_img = dataset + ann['image']
            image = cv.imread(path_img)

            path = 'grabcut_kaggle_dataset_folder/' + label + '/'
            print(os.path.join(path, img_name))

            cv.imwrite(os.path.join(path, img_name), image)

