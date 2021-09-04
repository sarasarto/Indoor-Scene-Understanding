import argparse

from retrieval.method_SIFT.helper_SIFT import SIFTHelper
from retrieval.method_autoencoder.helper_autoenc import AutoencHelper
from retrieval.method_dhash.helper_DHash import DHashHelper
from retrieval.query_expansion_transformations import QueryTransformer
from retrieval.retrieval_manager import ImageRetriever
from geometry.rectification2 import *
from classification.classification_utils import Classification_Helper
from plotting_utils.plotter import Plotter
from furniture_segmentation.prediction_model import PredictionModel
from PIL import Image
from geometry.rectification2 import GeometryRectification
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
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

    # img = cv2.blur(np.array(img),(5,5)) con questo non trova un risultato
    img = cv2.bilateralFilter(np.array(img), 9, 75, 75)

    transform = transforms.Compose([
    transforms.ToTensor()])
    img = transform(img)

except :
    raise ValueError('Impossible to open the specified file. Check the name and try again.')


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

print(f'Objects founded in the image: {text_labels}')

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

for bbox, label, mask in zip(boxes,text_labels, masks):
    # bbox is the query
    # for each method (SIFT, DHash, autoencoder) show results.
    bbox = list(map(int,np.round(bbox)))
    xmin = bbox[0]
    xmax = bbox[2]
    ymin = bbox[1]
    ymax = bbox[3]
    
    if label in retrieval_classes:

        # we use the result mask from the network in order to apply grabcut
        # all the pixels equal to 0 Grabcut considers them as "Probable_Background"
        # all the pixels equal to 1 are considered "Sure_Foreground" pixels
        mask = mask.astype('uint8')
        mask[mask==0] = 2

        query_img = img[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]
        #query_img = cv2.blur(np.array(query_img), (5, 5))
        #query_img = cv2.bilateralFilter(np.array(query_img), 9, 75, 75)
        #query_img = cv2.medianBlur(query_img,5)
        query_img = cv2.bilateralFilter(np.array(query_img), 9, 50, 50)

        # query_image = cv2.GaussianBlur(query_img,(5,5),0)
        #if necessario perche la rete ritorna pendant lamp e nel dataset retrieval(comprese annnotazioni)
        #abbiamo 'lamp'
        if 'lamp' in label:
            label = 'lamp'

        #query processing, application of grabcut and same other filters(yet to decide)
        res_img = qt.extract_query_foreground(query_img, mask) #the result is the query without background

        #pt.plot_imgs_by_row([query_img, res_img], ['Query img', 'Result with grabcut'], 2)
        #sift method
        # img_retriever = ImageRetriever(SIFTHelper())
        # sift_results = img_retriever.find_similar_furniture(res_img, label)
        # pt.plot_retrieval_results(query_img, sift_results, 'sift')

        #dhash method
        # img_retriever = ImageRetriever(DHashHelper())
        # PIL_image = Image.fromarray(np.uint8(res_img)).convert('RGB')
        # dhash_results = img_retriever.find_similar_furniture(PIL_image, label)
        # pt.plot_retrieval_results(query_img, dhash_results, 'dhash')


        # autoencoder method
        #img_retriever = ImageRetriever(AutoencHelper())
        #autoenc_results = img_retriever.find_similar_furniture(Image.fromarray(res_img), label)
        #pt.plot_retrieval_results(query_img, autoenc_results, 'autoencoder')

        

    # elif label in rectification_classes:
    #     query_img = img[ymin-10:ymax+10, xmin-10:xmax+10]
    #     #print(f'Trovata label: {label}')
    #     #rect = rectification(query_img)
    #     #io.imsave('geometry/result.png', rect.rectify_image(clip_factor=4, algorithm='independent'))
    #     #pass
    #     gr = GeometryRectification()
    #     gr.rectification(query_img)



#ROOM CLASSIFICATION PHASE
#construct vector
#i load the dataset info
classification_helper = Classification_Helper()
feature_vector = classification_helper.construct_fv_for_prediction(labels)
predicted_room, text_prediction = classification_helper.predict_room(feature_vector)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Prediction is {text_prediction}')
plt.show() 