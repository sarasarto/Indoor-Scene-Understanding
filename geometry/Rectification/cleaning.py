import cv2
import numpy as np


def closing(input: np.array, size=5, erode=True, debug=False):
    img = _closing(input, size, erode)
    if debug:
        return img, img
    else:
        return img


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


def invert(input: np.array, debug=False):
    result = _invert(input)
    if debug:
        return result, result
    else:
        return result


def _invert(img: np.array):
    """
    White becomes Black and viceversa
    ----------
    img : np.array
        image where to apply the inversion
    """
    return 255 - img


def erode_dilate_invert(img: np.array, size=5, erode=True):
    inversion = invert(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=np.ones((size, size), np.uint8), iterations=5))
    return inversion


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


def add_padding(input, pad=100, color=[0, 0, 0], debug=False):
    result = _add_padding(input, pad, color)
    if debug:
        return result, result
    else:
        return result