import cv2
import numpy as np


def _mean_shift_segmentation(img: np.array, spatial_radius=3, color_radius=35, maximum_pyramid_level=3):
    """
    This function takes an image and mean-shift parameters and
    returns a version of the image that has had mean shift
    segmentation performed on it

    See also: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#pyrmeanshiftfiltering
    Parameters
    ----------
    img : np.array
        image where to find the mean-shift segmentation
    spatial_radius : int
        The spatial window radius.
    color_radius : int
        The color window radius.
    maximum_pyramid_level : int
        Maximum level of the pyramid for the segmentation.
    """
    img = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius, maximum_pyramid_level)
    return img
