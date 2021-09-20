import cv2
import numpy as np

"""
    Cleans the noise around paintings' frames though erosion

    Parameters
    ----------
    img : the image to be cleaned
    Returns
    -------
    img : the cleaned image
"""


def _clean_frames_noise(img, k_size=23, iterations=1):
    kernel = np.ones((k_size, k_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening


def _mask_from_contour(img, contour):
    canvas = np.zeros_like(img)
    cv2.fillPoly(canvas, pts=[contour], color=(255, 255, 255))
    return canvas


"""
    Applies the proper median filter for smoothing the frame's sides
    Parameters
    ----------
    img : the input image
    Returns
    -------
    img : the smoothed image
"""


def _apply_median_filter(img, strength=15):
    result = cv2.medianBlur(img, strength)
    return result

"""
    Applies Canny's edge detector
    Parameters
    ----------
    img :  the input image
    Returns
    -------
    img : the image's edges
"""


def _apply_edge_detection(img, t1=50, t2=100):
    result = cv2.Canny(img, t1, t2)
    return result