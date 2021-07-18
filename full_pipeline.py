import argparse
from PIL import Image
import torchvision
import torch
import json
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import random as rng
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import colorsys
import cv2
from scipy import ndimage
import random
from geometric_transformations.geometric_transformations import GeometryTransformer
from classification.classification_utils import Classification_Helper
    
def random_colors(N, brightness=0.01):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
   
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print(f'in_features: {in_features}')
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    print(f'in_features_mask: {in_features_mask}')
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def show_bboxes_and_masks(image, boxes, masks, colors, alpha=0.2):
    for i,mask, bbox in zip(range(len(masks)), masks, boxes):
        #define bbox
     
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1], fill=False,
            edgecolor='blue', linewidth=2, alpha=0.9)
        )

        plt.gca().text(bbox[0], bbox[1] - 2,
            '%s' % ('ciao'),
            bbox=dict(facecolor='blue', alpha=0.2),
            fontsize=10, color='white')

        #define mask
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *
                                    (1 - alpha) + alpha * colors[i][c] * 255,
                                    image[:, :, c])
    plt.imshow(image)
    plt.show()


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


#now I can use my model to segment the image(for the moment we use a fully pretrained maskrcnn)
#model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = get_instance_segmentation_model(102)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu'))['model_state_dict'])
model.eval()

prediction = model([img])


scores = prediction[0]['scores']
scores = scores > 0.7
scores = scores.nonzero()
num_objs = len(scores)

boxes = prediction[0]['boxes'][:num_objs,:]
labels = prediction[0]['labels'][:num_objs]

with open('dataset_processing/mapping.json', 'r') as f:
    data = json.load(f)

#funtion to make the mapping in text
text_labels = []
for label in labels:
    for key in data:
        if data[key]['new_label'] == label:
            text_labels.append(key)


boxes = boxes.detach().numpy()
masks = prediction[0]['masks'][:num_objs,:,:]
masks = masks.detach().numpy().round()
masks = np.squeeze(masks)


img = img.detach().numpy()
img = np.swapaxes(img, 0,2)
img = np.swapaxes(img, 0,1)

#show_bboxes_and_masks(img, boxes, masks, random_colors(num_objs))

#parte di processing prima di estrarre i keypoint per il retrieval
gt = GeometryTransformer()

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

for box, mask, text_label in zip(boxes, masks, text_labels):
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
        
        #cropped_image[dst>0.01*dst.max()]=[0,0,255]
        coordinates = np.argwhere(dst > 0.01*dst.max())

        top = coordinates[0]
        bottom = coordinates[-1]
        left = coordinates[np.argmin(coordinates[:,1])]
        right = coordinates[np.argmax(coordinates[:,1])]

        corners = np.array([top,
                            bottom,
                            left,
                            right])

        #tl = (np.min(coordinates[:,0]), np.max(coordinates[:,1]))
        #tr = (np.max(coordinates[:,0]), np.max(coordinates[:,1]))
        #br = (np.max(coordinates[:,0]), np.min(coordinates[:,1]))
        #bl = (np.min(coordinates[:,0]), np.min(coordinates[:,1]))
        print(coordinates.shape)
        print(coordinates)
        plt.imshow(cropped_image)
        plt.show()

        x = [top[1], bottom[1], left[1],right[1]]
        #print(x)
        y = [top[0], bottom[0], left[0],right[0]]
        print(x,y)

        #test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #print(kp1[0].pt)
        implot = plt.imshow(cropped_image)
        plt.scatter(x,y, c='r')
        plt.show()

        rectified_furniture = gt.furniture_rectification(cropped_image, corners)
        plt.imshow(rectified_furniture)
        plt.show()



        '''
        rotated_image = ndimage.rotate(cropped_image, 45)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(rotated_image,None)
        test = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ndimage.rotate(cropped_image, -45),None)
        test = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''

        '''
        plt.imshow(cropped_image)
        plt.show()
        src_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        plt.imshow(src_gray, cmap='gray')
        plt.show()
        src_gray = np.round(src_gray).astype(np.uint8)
        src_gray = np.where(src_gray==1, 255, 0).astype(np.uint8)
  
        
        fig, axs = plt.subplots(1,2)
        axs[0,0].imshow(box)
        axs[0,1].imshhow(box)
        axs.show()
        
        canny_output = cv2.Canny(src_gray, 30, 50 * 3)
        plt.imshow(canny_output, cmap='gray')
        plt.show()
        '''





#classification phase
#construct vector
#i load the dataset info
with open('dataset_processing/dataset_info_all_objs.json', 'r') as f:
    data = json.load(f)

with open('dataset_processing/mapping.json', 'r') as f:
    mapping = json.load(f)

num_objs = len(data['instances_per_obj'])
vector = np.zeros((1,num_objs))
labels = np.unique(labels) #-1 because labels start from 1 but array indexing from 0
idxs = []
for label in labels:
    for map in mapping:
        if mapping[map]['new_label'] == label:
            print(map)
            idxs.append(list(data['instances_per_obj']).index(map))
            print(list(data['instances_per_obj']).index(map))

vector[:,idxs] = 1
classification_helper = Classification_Helper()
predicted_room = classification_helper.predict_room(vector)

print(f'The predicted room is: {predicted_room}')





