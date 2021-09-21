import cv2
import numpy as np


def _closing(img: np.array, size=20, erode=True):
    """
    Closing the image
    ----------
    img : np.array
        image where to apply the closing
    """
    kernel = np.ones((size, size), np.uint8)
    img = cv2.dilate(img, kernel)
    if erode:
        img = cv2.erode(img, kernel)
    return img


def _clean_frames_noise(img, k_size=23, iterations=1):
    """
        Cleans the noise

        Parameters
        ----------
        img : the image to be cleaned
        Returns
        -------
        img : the cleaned image
    """
    kernel = np.ones((k_size, k_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening