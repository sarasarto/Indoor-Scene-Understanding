import argparse
from classification.classification_utils import Classification_Helper
from plotting_utils.plotter import Plotter
from furniture_segmentation.prediction_model import PredictionModel
from furniture_segmentation.training_utils import get_instance_model_default, get_instance_model_modified
from PIL import Image
import torch
from torchvision.transforms import transforms
import numpy as np
import cv2
import json
from geometric_transformations.geometric_transformations import GeometryTransformer


parser = argparse.ArgumentParser(description='Computer Vision pipeline')
parser.add_argument('-img', '--image', type=str,
                    help='path of the image to analyze', required=True)

args = parser.parse_args()
img_path = args.image

try:
    img = Image.open(img_path)
    transform = transforms.Compose([                       
    transforms.ToTensor()])
    img = transform(img)
except FileNotFoundError:
    print('Impossible to open the specified file. Check the name and try again.')

num_classes = 1324
pm = PredictionModel('model_mask_modified.pt', num_classes, default_model=False)
prediction = pm.segment_image(img)
boxes, masks, labels, scores = pm.extract_furniture(prediction, 0.8)

with open('ADE20K_filtering/_old_mapping.json', 'r') as f:
    data = json.load(f)

#funtion to make the mapping in text
text_labels = []
for label in labels:
    for key in data:
        if data[key]['new_label'] == label:
            text_labels.append(key)


img = cv2.imread(img_path)
pt = Plotter()
pt.show_bboxes_and_masks(img, boxes, masks, text_labels, scores)
 


for i in boxes:
    # il box è la mia query
    # 
    pass


'''
#parte di processing prima di estrarre i keypoint per il retrieval
gt = GeometryTransformer()

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


for box, mask, text_label in zip(boxes, masks, text_labels):
    print(text_label)
    print(mask.shape)
    print(np.unique(mask))
    #result = gt.transform_bbox(box)
    if text_label == 'sofa':
        #box = np.round(box)
        box = box.astype(int)
        cropped_image = img[box[1]:box[3], box[0]:box[2]]
        cropped_mask = mask[box[1]:box[3], box[0]:box[2]]
   
        #plt.imshow(cropped_image)
        #plt.show()
        src_gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()

        src_gray = np.round(src_gray).astype(np.uint8)
        kp1, des1 = sift.detectAndCompute(src_gray,None)
        test = cv2.drawKeypoints(src_gray, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #plt.imshow(test)
        #plt.show()

        canny_output = cv2.Canny(src_gray, 30, 50 * 3)
        #plt.imshow(canny_output, cmap='gray')
        #plt.show()

        dst = cv2.cornerHarris(src_gray,2,3,0.04)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        
        
        coordinates = np.argwhere(dst > 0.01*dst.max())
    
        mask_x, mask_y = np.where(cropped_mask==1)
        print(mask_x.shape)
        mask_x = mask_x.reshape((-1,1))
        print(mask_x.shape)
        mask_y = mask_y.reshape((-1,1))
        mask_points_coord = np.hstack((mask_x, mask_y))
        coordinates[:, [1, 0]] = coordinates[:, [0, 1]]
        mask_points_coord[:, [1, 0]] = mask_points_coord[:, [0, 1]]
        print(coordinates)
        print(mask_points_coord)


        aset = set([tuple(x) for x in coordinates])
        bset = set([tuple(x) for x in mask_points_coord])
        intersection = [x for x in aset & bset]
        #print('iniziatooo')
        #print(intersection)

        for item_a in aset:
            x_a, y_a = item_a[0], item_a[1]
            for item_b in bset:
                x_b, y_b = item_b[0], item_b[1]
                if x_a - 5 <= x_b <= x_a + 5 and y_a - 5<= y_b <= y_a + 5:
                    intersection.append((x_b, y_b))


        #nb trasformare in array
        intersection = np.array(intersection)
        #intersection = intersection[np.lexsort(intersection[:,::-1]).T]
        #intersection[:, [1, 0]] = intersection[:, [0, 1]]

        print(intersection)
     
        top = coordinates[coordinates[:,1].argmin()]
        bottom = coordinates[coordinates[:,1].argmax()]
        left = coordinates[np.argmin(coordinates[:,0])]
        right = coordinates[np.argmax(coordinates[:,0])]

        x = [left[0], bottom[0], right[0], top[0]]
        y = [left[1], bottom[1], right[1], top[1]]
        print(x, y)

        plt.imshow(cropped_image)
        plt.scatter(x, y, c='r')
        plt.show()


        top = mask_points_coord[mask_points_coord[:,1].argmin()]
        bottom = mask_points_coord[mask_points_coord[:,1].argmax()]
        left = mask_points_coord[mask_points_coord[:, 0].argmin()]
        right = mask_points_coord[(mask_points_coord[:,0]).argmax()]

        x = [left[0], bottom[0], right[0], top[0]]
        y = [left[1], bottom[1], right[1], top[1]]
        print(x, y)

        plt.imshow(cropped_image)
        plt.scatter(x, y, c='r')
        plt.show()

      
        #tl, bl, br, tr
        corners = np.array([[1,10],
                            [6,92],
                            [275,105],
                            [289,9]])
''' 

#classification phase
#construct vector
#i load the dataset info
classification_helper = Classification_Helper()
feature_vector = classification_helper.construct_fv_for_prediction(labels)
predicted_room, text_prediction = classification_helper.predict_room(feature_vector)
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Prediction is {text_prediction}')
plt.show()




