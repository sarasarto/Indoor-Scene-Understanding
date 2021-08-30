import argparse
from retrieval.method_dhash.helper_DHash import DHashHelper
from retrieval.method_SIFT.helper_SIFT import SIFTHelper
from retrieval.method_autoencoder.helper_autoenc import AutoencHelper
from retrieval.query_expansion_transformations import QueryTransformer
from retrieval.retrieval_manager import ImageRetriever
from classification.classification_utils import Classification_Helper
from plotting_utils.plotter import Plotter
from furniture_segmentation.prediction_model import PredictionModel
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import cv2
import json

parser = argparse.ArgumentParser(description='Computer Vision pipeline')
parser.add_argument('-img', '--image', type=str,
                    help='path of the image to analyze', required=True)
parser.add_argument('-mdl', '--model', type=str,
                    help='type of model (default or modified)', required=True)

args = parser.parse_args()
img_path = args.image
model_type = args.model

if model_type not in ['default', 'modified']:
    raise ValueError('Model type must be \'default\' or \'modified\'')

try:
    img = Image.open(img_path)
    transform = transforms.Compose([                       
    transforms.ToTensor()])
    img = transform(img)
except FileNotFoundError:
    print('Impossible to open the specified file. Check the name and try again.')

#-------------------------------------------------------SEGMENTATION PHASE--------------------------------------------------------#
num_classes = 1324 #1323 classes + 1 for background

if model_type == 'default':
    PATH = 'model_mask_default.pt'
    is_default = True
else:
    PATH = 'model_mask_default.pt'
    is_default = False

pm = PredictionModel(PATH, num_classes, is_default)
prediction = pm.segment_image(img)
boxes, masks, labels, scores = pm.extract_furniture(prediction, 0.7)

with open('ADE20K_filtering/filtered_dataset_info.json', 'r') as f:
    data = json.load(f)

text_labels = []
for label in labels:
    for key in data['objects']:
        if data['objects'][key]['new_label'] == label:
            text_labels.append(data['objects'][key]['labels'][0])


img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
pt = Plotter()
pt.show_bboxes_and_masks(img, boxes, masks, text_labels, scores) 


#------------------------------------------------------------RETRIEVAL PHASE----------------------------------------------#
retrieval_classes = []
with open('retrieval/retrieval_classes.txt') as f:
    retrieval_classes = f.read().splitlines()

rectification_classes = []
with open('geometry/objects_for_rectification.txt') as f:
    rectification_classes = f.read().splitlines()

qt = QueryTransformer()

for bbox, label in zip(boxes,text_labels):
    # bbox is the query
    #for each method (SIFT, DHash, autoencoder) show results.

    if label in retrieval_classes:
        bbox = list(map(int,np.round(bbox)))
        xmin = bbox[0]
        xmax = bbox[2]
        ymin = bbox[1]
        ymax = bbox[3]
        
        query_img = img[ymin:ymax, xmin:xmax]
        #query processing, application of grabcut and same other filters(yet to decide)
        res_img = qt.extract_query_foreground(query_img) #the result is the query without background

        #NB: IL GRABCUT PUO' DAVVERO FUNZIONARE? MIGLIORA DAVVERO LE PRESTAZIONI?
        #CMQ LO LASCIAMO PER FAR VEDERE CHE ABBIAMO FATTO QUALCOSA IN PIU'
        pt.plot_imgs_by_row([query_img, res_img], ['Query img', 'Result with grabcut'], 2)
  
        #sift method
        img_retriever = ImageRetriever(SIFTHelper())
        sift_results = img_retriever.find_similar_furniture(res_img, label)
        pt.plot_retrieval_results(query_img, sift_results)

        #dhash method
        #NB: L'ATTUALE IMPLEMENTAZIONE PREVEDE CHE SI RICALCOLI L'HASH DEL DATASET PER OGNI QUERY.
        #IN ALTERNATIVA(FORSE SARABBE MEGLIO) SAREBBE SALVARSI IN QUALCHE MODO L'HASH DEL DATASET.
        img_retriever = ImageRetriever(DHashHelper())
        dhash_results = img_retriever.find_similar_furniture(query_img, label)
        pt.plot_retrieval_results(query_img, dhash_results)

        #autoencoder method
        img_retriever = ImageRetriever(AutoencHelper())
        autoenc_results = img_retriever.find_similar_furniture(query_img, label)
        
        #axarr[i].imshow(img[ymin:ymax, xmin:xmax])
        #i += 1

    elif label in rectification_classes:
        #do rectification (kevin code must be inserted)
        pass
plt.show()


#RECTIFICATION PHASE
...



#ROOM CLASSIFICATION PHASE
#construct vector
#i load the dataset info
classification_helper = Classification_Helper()
feature_vector = classification_helper.construct_fv_for_prediction(labels)
predicted_room, text_prediction = classification_helper.predict_room(feature_vector)
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Prediction is {text_prediction}')
plt.show()

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