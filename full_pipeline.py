import argparse
from furniture_segmentation.training_utils import get_instance_model_default, get_instance_model_modified
from PIL import Image
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import random
from geometric_transformations.geometric_transformations import GeometryTransformer

    
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
model = get_instance_model_default(102)
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu'))['model_state_dict'])
model.eval()

prediction = model([img])

'''
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
if len(masks) > 1:
    masks = np.squeeze(masks)
else:
    masks = np.squeeze(masks, axis=0)
    print(masks.shape)


img = img.detach().numpy()
img = np.swapaxes(img, 0,2)
img = np.swapaxes(img, 0,1)

show_bboxes_and_masks(img, boxes, masks, random_colors(num_objs))

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
'''




