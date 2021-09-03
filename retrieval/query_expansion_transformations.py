import cv2
import matplotlib.pyplot as plt
import numpy as np

class QueryTransformer():
    def __init__(self):
        pass
        
    def rotate_image(self, image, rotation_angle=0):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rotation_angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def flip_image(self, image, flip_code):
        #flip_code: 0=flip_vertical, 1=flip_horizontal, -1=flip_both
        flipped_img = cv2.flip(image, flip_code)
        return flipped_img

    def scale_img(self, image, scale_factor):
        H = image.shape[0]
        W = image.shape[1]
        dim = ((int)(W*scale_factor), (int)(H*scale_factor))
        resized_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized_img
    
    
    def extract_query_foreground(self, image, mask):
        #height, width = image.shape[0], image.shape[1]
        #rect = (1, 1, width, height)
        #mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]

        return image

