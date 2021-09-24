import argparse
import torch
from classification.MLP_model import HomeScenesClassifier
from retrieval.method_SIFT.helper_SIFT import SIFTHelper
from retrieval.method_autoencoder.helper_autoenc import AutoencHelper
from retrieval.method_dhash.helper_DHash import DHashHelper
from retrieval.query_expansion_transformations import QueryTransformer
from retrieval.retrieval_manager import ImageRetriever
from classification.classification_utils import Classification_Helper
from plotting_utils.plotter import Plotter
from furniture_segmentation.prediction_model import PredictionModel
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import cv2
from geometry.Rectification.image_rectification import ImageRectifier
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Computer Vision pipeline')
parser.add_argument('-img', '--image', type=str,
                    help='path of the image to analyze', required=True)
parser.add_argument('-mdl', '--model', type=str,
                    help='type of model (default or modified)', required=True)
parser.add_argument('-rtv', '--retrieval', type=str,
                    help='retrieval method to use(sift, dhash or autoencoder)', required=True)
parser.add_argument('-clf', '--classifier', type=str,
                    help='type of classification model (forest or mlp)', required=True)

args = parser.parse_args()
img_path = args.image
model_type = args.model
retr_type = args.retrieval
clf_mode = args.classifier

if model_type not in ['default', 'modified']:
    raise ValueError('Model type must be \'default\' or \'modified\'')

if clf_mode not in ['forest', 'mlp']:
    raise ValueError('Classifier be \'forest\' or \'mlp\'')

if retr_type not in ['sift', 'dhash', 'autoencoder']:
    raise ValueError('Retrieval type must be \'sift\', \'dhash\' or \'autoencoder\'')

try:
    img = Image.open(img_path)
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
    PATH = 'model_mask_modified.pt'
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

        # # we use the result mask from the network in order to apply grabcut
        # # all the pixels equal to 0 Grabcut considers them as "Probable_Background"
        # # all the pixels equal to 1 are considered "Sure_Foreground" pixels
        mask = mask.astype('uint8')
        mask[mask==0] = 2
        
        query_img = img[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]
        
        query_img = cv2.bilateralFilter(np.array(query_img), 9, 75, 75)
       
        if 'lamp' in label:
            label = 'lamp'
    
        res_img = qt.extract_query_foreground(query_img, mask) #the result is the query without background
        pt.plot_imgs_by_row([query_img, res_img], ['Query img', 'Result with grabcut'], 2)
    
        if retr_type == 'sift':
            img_retriever = ImageRetriever(SIFTHelper())
            sift_results = img_retriever.find_similar_furniture(res_img, label)
            pt.plot_retrieval_results(query_img, sift_results, 'sift')

        elif retr_type == 'dhash':
            img_retriever = ImageRetriever(DHashHelper())
            PIL_image = Image.fromarray(np.uint8(res_img)).convert('RGB')
            dhash_results = img_retriever.find_similar_furniture(PIL_image, label)
            pt.plot_retrieval_results(query_img, dhash_results, 'dhash')

        else:
            img_retriever = ImageRetriever(AutoencHelper())
            autoenc_results = img_retriever.find_similar_furniture(Image.fromarray(res_img), label)
            pt.plot_retrieval_results(query_img, autoenc_results, 'autoencoder')
      
    elif label in rectification_classes:
            query_img = img[ymin-10:ymax+10, xmin-10:xmax+10]

            print(f'Trovata label: {label}')
            img_rectifier = ImageRectifier()
            rect = img_rectifier.rectify(query_img)

            #print result of the rectification step  if it works correctly
            if rect is not None:
                pt.plot_imgs_by_row([query_img, rect], ['Extracted object', 'Rectified object'], 2)

  
#-----------------------------------------------------ROOM CLASSIFICATION PHASE-------------------------------------------------------
classification_helper = Classification_Helper()
feature_vector = classification_helper.construct_fv_for_prediction(labels)

if clf_mode == 'forest':
    predicted_room, text_prediction = classification_helper.predict_room(feature_vector)
else:
    model = HomeScenesClassifier(len(data['objects']))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load('classification/MLP_model.pt', map_location=device))
    model.eval()
    feature_vector = torch.tensor(feature_vector).type(torch.FloatTensor)
    result = model(feature_vector)
    predicted_class = torch.argmax(result)
    text_prediction = classification_helper.class2text_lbel(predicted_class)
    pt.plot_image(img, f'Prediction is {text_prediction}')