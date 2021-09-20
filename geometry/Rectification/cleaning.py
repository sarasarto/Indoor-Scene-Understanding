import cv2
import numpy as np


def _closing(img: np.array, size=20, erode=True):
    """
    Closing an image by using a specific structuring element
    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#dilate
    ----------
    img : np.array
        image where to apply the closing
    """
    kernel = np.ones((size, size), np.uint8)
    img = cv2.dilate(img, kernel)
    if erode:
        img = cv2.erode(img, kernel)
    return img


def _invert(img: np.array):
    """
    White becomes Black and viceversa
    ----------
    img : np.array
        image where to apply the inversion
    """
    return 255 - img


def _add_padding(img, pad=100, color=[0, 0, 0]):
    result = cv2.copyMakeBorder(
        img,
        top=pad,
        bottom=pad,
        left=pad,
        right=pad,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return result