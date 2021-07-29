import cv2
import numpy as np
import matplotlib.pyplot as plt

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
