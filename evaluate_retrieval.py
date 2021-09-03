import argparse
from retrieval.query_expansion_transformations import QueryTransformer
from furniture_segmentation.prediction_model import PredictionModel
import os
from torchvision.transforms import transforms
import numpy as np
import cv2
import json
from plotting_utils.plotter import Plotter
from retrieval.retrieval_manager import ImageRetriever
from retrieval.method_dhash.helper_DHash import DHashHelper
from retrieval.method_SIFT.helper_SIFT import SIFTHelper
from retrieval.method_autoencoder.helper_autoenc import AutoencHelper
from evaluation.eval_manager import Evaluator
from PIL import Image

parser = argparse.ArgumentParser(description='Retrieval Evaluation')
parser.add_argument('-mdl', '--model', type=str,
                    help='type of model (default or modified)', required=True)
parser.add_argument('-retr', '--retrieval', type=str,
                    help='Type of retrieval method (sift/ dhash / autoencoder)', required=True)

args = parser.parse_args()
model_type = args.model
retr_method = args.retrieval

if model_type not in ['default', 'modified']:
    raise ValueError('Model type must be \'default\' or \'modified\'')
if retr_method not in ['sift', 'dhash', 'autoencoder']:
    raise ValueError('Model type must be \'sift\' or \'dhash\' or \'autoencoder\'')

if retr_method == 'sift':
    helper = SIFTHelper()
else:
    if retr_method == 'dhash':
        helper = DHashHelper()
    else:
        helper = AutoencHelper()

retrieval_classes = []
with open('retrieval/retrieval_classes.txt') as f:
    retrieval_classes = f.read().splitlines()

# folder con test images da ADE20
test_image = 'test_retrieval'
AP_vector = []

for img in os.listdir(test_image):

    print("Evaluating New Image...")
    img_path = test_image + '/' + img
    try:
        img = Image.open(img_path)
        transform = transforms.Compose([
            transforms.ToTensor()])
        img = transform(img)

        img_retriever = ImageRetriever(helper)

        # -------------------------------------------------------SEGMENTATION PHASE--------------------------------------------------------#
        num_classes = 1324  # 1323 classes + 1 for background

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

        # ------------------------------------------------------------RETRIEVAL PHASE----------------------------------------------#
        retrieval_classes = []
        with open('retrieval/retrieval_classes.txt') as f:
            retrieval_classes = f.read().splitlines()

        rectification_classes = []
        with open('geometry/objects_for_rectification.txt') as f:
            rectification_classes = f.read().splitlines()

        qt = QueryTransformer()

        for bbox, label in zip(boxes, text_labels):
            # bbox is the query
            # for each method (SIFT, DHash, autoencoder) show results.
            bbox = list(map(int, np.round(bbox)))
            xmin = bbox[0]
            xmax = bbox[2]
            ymin = bbox[1]
            ymax = bbox[3]

            if label in retrieval_classes:
                query_img = img[ymin:ymax, xmin:xmax]
                # if necessario perche la rete ritorna pendant lamp e nel dataset retrieval(comprese annnotazioni)
                # abbiamo 'lamp'
                if 'lamp' in label:
                    label = 'lamp'

                # query processing, application of grabcut and same other filters(yet to decide)
                res_img = qt.extract_query_foreground(query_img)  # the result is the query without background
                pt.plot_imgs_by_row([query_img, res_img], ['Query img', 'Result with grabcut'], 2)

                if retr_method == 'sift':
                    results = img_retriever.find_similar_furniture(res_img, label)
                if retr_method == 'dhash':
                    PIL_image = Image.fromarray(np.uint8(res_img)).convert('RGB')
                    results = img_retriever.find_similar_furniture(PIL_image, label)
                if retr_method == 'autoencoder':
                    results = img_retriever.find_similar_furniture(Image.fromarray(res_img), label)

                pt = Plotter()
                pt.plot_retrieval_results(query_img, results, retr_method)
                img_evaluator = Evaluator(query_img, results)
                single_AP = img_evaluator.eval()
                AP_vector.append((single_AP))
                print("AP vector:")
                print(AP_vector)



    except:
        raise ValueError('Impossible to open the specified file. Check the name and try again.')

print("computing MAP on test kaggle dataset ")
print(np.mean(AP_vector))
