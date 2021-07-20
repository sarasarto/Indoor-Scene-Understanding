import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import torch
import torchvision
import PIL
import random
import os
import json


#uso per mask-r-cnn
img = '/Users/kevinmarchesini/Indoor-Scene-Understanding/bedroom.jpg'
matplotlib.image.imread(img)
img = PIL.Image.open(img).convert('RGB')
image_tensor = torchvision.transforms.functional.to_tensor(img)

torch.set_grad_enabled(False)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.eval()

output = model([image_tensor])[0]
print(output['boxes'])
print(output['labels'])
print(output['scores'])

coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
              'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
              'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
              'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
              'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender',
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]
result_image = np.array(img.copy())

for i, c in enumerate(coco_names): print(i,c)
for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
    if coco_names[label] == 'bed' and score > 0.7:
        image = cv.imread('/Users/kevinmarchesini/Indoor-Scene-Understanding/bedroom.jpg')
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        x, y, width, height = int(box[0]), int(box[1]), int(box[2])-int(box[0]), int(box[3])-int(box[1])
        rect = (x, y, width, height)
        cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]
        crop_img = image[y:y + height, x:x + width]
        plt.imshow(crop_img), plt.colorbar(), plt.show()

        color = random.choice(colors)

        # draw box
        tl = round(0.002 * max(result_image.shape[0:2])) + 1  # line thickness
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        area = (c1[0] - c2[0]) * (c1[1] - c2[1])

        cv.rectangle(result_image, c1, c2, color, thickness=tl)
        # draw text
        display_txt = "%s: %.1f%%" % (coco_names[label], 100 * score)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(result_image, c1, c2, color, -1)  # filled
        cv.putText(result_image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                    lineType=cv.LINE_AA)


# code to see the result of mask-r-cnn to detect the bounding box
plt.figure(figsize=(45, 15))
plt.imshow(result_image)
plt.show()


data = json.load(open("Annotations.json"))

# immagine di riferimento. Vorrei tornare immagini simili a lei

gray_l = cv.cvtColor(crop_img, cv.COLOR_RGB2GRAY)
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_l, None)
test = cv.drawKeypoints(crop_img, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(test)
plt.title("keypoints arredo immagine principale")
plt.show()

# confronto tutte le immagini della stessa categoria
num_good = []

obj_list = []
path = '/Users/kevinmarchesini/Documents/Retrieval_SIFT/retrieval_grabcut'
class_retrieval = "bed"
for im in data:
    if im["annotations"][0]["label"] == class_retrieval:
        #print(im["image"])
        obj_list.append(im["image"])
        path2 = os.path.join(path, im["image"])
        img2 = cv.imread(path2, cv.COLOR_BGR2RGB)
        gray_2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)


        sift = cv.SIFT_create()
        kp2, des2 = sift.detectAndCompute(gray_2, None)
        test2 = cv.drawKeypoints(img2, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #plt.imshow(test2)
        #plt.show()

        # cv2.BFMatcher() takes the descriptor of one feature in first set
        # and is matched with all other features in second set using some distance calculation.
        # And the closest one is returned.

        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # print("match trovati: " + str(len(matches)))

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        num_good.append(len(good))
        # print("match buoni trovati: " + str(len(good)))
        img3 = cv.drawMatchesKnn(crop_img, kp1, img2, kp2, good, None, flags=2)
        #plt.imshow(img3), plt.show()

# stampo i punti che soddisfano la percentuale per ogni immagine
for i, img in enumerate(obj_list):
    print(str(img) + '\t' + str(num_good[i]))
num_good = np.array(num_good)

# decido di tenere solo i tre migliori
num_good_sorted = num_good.argsort()

best = []
for i, img in enumerate(obj_list):
    if i in num_good_sorted[-3:]:
        path = '/Users/kevinmarchesini/Documents/RetrievalDataset/downloads/' + class_retrieval + '/'
        best.append(str(img))
        path = os.path.join(path, img)
        img = cv.imread(path, cv.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

print("Le migliori tre corrispondenze: " + str(best))

