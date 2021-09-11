from __future__ import print_function
import time
import os
import cv2 as cv
import matplotlib
import matplotlib.pylab as pltpy
import random
import PIL
import torchvision
import numpy as np
import torch
import random as rng
import matplotlib.pyplot as plt
from plotting_utils.plotter import Plotter

annotations = []


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 3)

    pl.plot_image(cv.cvtColor(canny_output, cv.COLOR_BGR2RGB))

    # function findsCounter():
    # RETR_EXTERNAL:only retrieve the outermost contour;
    # RETR_TREE (most commonly used): retrieve all contours and reconstruct the entire hierarchy of nested contours;

    # CHAIN_APPROX_NONE: The outline is output in the form of Freeman chain code, and all other methods output polygons (sequence of vertices).
    # CHAIN_APPROX_SIMPLE (most commonly used): Compress the horizontal, vertical and diagonal parts, that is, the function only retains their end parts.

    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # the coordinates are returned in this way: (x, y , w , h)
    # (x,y) is the upper left point
    # w is width, h is height
    # for the area we just need base*height --> w* h
    print("coordinate dei box")
    print(boundRect)

    area = []
    for b in range(len(boundRect)):
        # Area
        area.append(boundRect[b][2] * boundRect[b][3])

    print("area ")
    print(area)

    # the biggest area is taken

    max = np.argmax(area)

    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.drawContours(drawing, contours_poly, max, color)
    cv.rectangle(drawing, (int(boundRect[max][0]), int(boundRect[max][1])), \
                 (int(boundRect[max][0] + boundRect[max][2]), int(boundRect[max][1] + boundRect[max][3])), color, 5)

    # cv.imshow('Contours', drawing)
    # cv.waitKey()
    pl.plot_image(drawing)
    x = boundRect[max][0]
    y = boundRect[max][1]
    width = boundRect[max][2]
    height = boundRect[max][3]

    return area[max], x, y, width, height


# to generate an annotation af an image
def generate_json(file_name, name_class, x, y, width, height):
    image_dict = {"image": '', "annotations": []}
    label_dict = {"label": '', "coordinates": {}}
    coord_dict = {"x": int, "y": int, "width": int, "height": int}

    coord_dict['x'] = x
    coord_dict['y'] = y
    coord_dict['width'] = width
    coord_dict['height'] = height

    label_dict['label'] = name_class
    label_dict['coordinates'] = coord_dict

    image_dict['image'] = file_name
    image_dict['annotations'].append(label_dict)

    annotations.append(image_dict)


if __name__ == '__main__':

    # creates the dataset and the annotations
    # these are the retrieval class
    retr_class = ['chair','lamp', 'sofa', 'armchair', 'chair', 'bed', 'bicycle']
    # c_class says if a class is present in COCO dataset or if it has a different name
    c_class = ['chair', 'None', 'couch', 'None', 'chair', 'bed', 'bicycle']

    for name, coco_name in zip(retr_class, c_class):
        images = []
        folder = 'dataset_retrieval_folder/' + name
        for filename in os.listdir(folder):
            if filename != '.DS_Store':
                # filename = folder + '/' + filename
                images.append(filename)
        print(images)
        torch.set_grad_enabled(False)
        pltpy.rcParams["axes.grid"] = False
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model = model.eval()
        pl = Plotter()

        # if it is a category of coco dataset, thus the bounding box is found with the pre-trained mask-r-cnn
        if coco_name != 'None':
            print("è di coco!!")
            for i, image in enumerate(images):
                print(image)
                image = folder + '/' + image
                matplotlib.image.imread(image)
                t = time.time()
                image = PIL.Image.open(image).convert('RGB')
                image_tensor = torchvision.transforms.functional.to_tensor(image)
                output = model([image_tensor])[0]
                print('executed in %.3fs' % (time.time() - t))
                print(output['boxes'])
                print(output['labels'])
                print(output['scores'])

                coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                              'truck', 'boat',
                              'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench',
                              'bird', 'cat',
                              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
                              'umbrella',
                              'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                              'sports ball',
                              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                              'bottle',
                              'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                              'sandwich',
                              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                              'potted plant',
                              'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop',
                              'mouse',
                              'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                              'refrigerator', 'blender',
                              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]

                result_image = np.array(image.copy())
                valid_image = False
                for box, label, score in zip(output['boxes'], output['labels'], output['scores']):

                    if score > 0.5:
                        color = random.choice(colors)

                        # draw box
                        tl = round(0.002 * max(result_image.shape[0:2])) + 1  # line thickness
                        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        area = (c1[0] - c2[0]) * (c1[1] - c2[1])
                        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(5)]

                        # I put the image in my dataset only if represent mainly the object that I want
                        if area > 120000 and coco_names[label] == coco_name:
                            valid_image = True
                            mask = np.zeros(result_image.shape[:2], np.uint8)
                            bgdModel = np.zeros((1, 65), np.float64)
                            fgdModel = np.zeros((1, 65), np.float64)
                            # name_class = coco_names[label]
                            x = max(0, c1[0] - 20)
                            y = max(0, c1[1] - 20)
                            width = min(result_image.shape[0] - c1[0], c2[0] - c1[0] + 40)
                            height = min(result_image.shape[1] - c1[1], c2[1] - c1[1] + 40)

                            # plotting the rectangle on the image
                            tl = 5 # line thickness
                            cv.rectangle(np.asarray(result_image), c1, c2, (255, 0, 0), tl)
                            pl.plot_image(image)
                            pl.plot_image(result_image)
                            generate_json(images[i], name, x, y, width, height)
                            break

                if valid_image == False:
                    os.remove(folder + '/' + images[i])

        # if it is not a category of coco dataset the bounding box is found with canny
        else:
            print("non una categoria di coco")
            for i, image in enumerate(images):
                print(image)
                image = folder + '/' + image
                src = cv.imread(image)
                pl.plot_image(plt.imread(image))

                # Convert image to gray and blur it
                src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
                src_gray = cv.blur(src_gray, (3, 3))
                source_window = 'Source'

                max_thresh = 255
                thresh = 50  # initial threshold

                # cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
                area, x, y, width, height = thresh_callback(thresh)

                # cv.waitKey()
                if area > 120000:
                    generate_json(images[i], name, x, y, width, height)
                else:
                    os.remove(folder + '/' + images[i])

    # create a json file with the annotations of each image and the corresponding bounding boxes
    # json_file = json.dumps(annotations)
    # with open('retrieval/Annotations_Kaggle.json', 'a') as f:
    #     f.write(json_file)
