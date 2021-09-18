import cv2
import numpy as np


# FLOORFILL
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill


def mask_largest_segment(input: np.array, debug=False, **kwargs):
    wallmask = _mask_largest_segment(input, **kwargs)

    if debug:
        return wallmask, wallmask
    else:
        return wallmask


def _mask_largest_segment(im: np.array, color_difference, scale_percent=1.0, x_samples=2, no_skip_white=False):
    """
    The largest segment will be white and the rest is black
    Useful to return a version of the image where the wall
    is white and the rest of the image is black.
    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill
    ----------
    img : np.array
        image where to find the largest element
    color_difference : int
        The distance from colors to permit.
    x_samples : int
        numer of samples that will be tested orizontally in the image
    """
    im = im.copy()

    h = im.shape[0]
    w = im.shape[1]
    color_difference = (color_difference,) * 3

    # in that way for smaller images the stride will be lower
    stride = int(w / x_samples)
    # --> DAVA PROBLEMI PER STRIDE A ZERO !!!!!!

    mask = np.zeros((im.shape[0] + 2, im.shape[1] + 2), dtype=np.uint8)
    wallmask = mask[1:-1, 1:-1].copy()
    largest_segment = 0
    for y in range(0, im.shape[0], stride):
        for x in range(0, im.shape[1], stride):
            if mask[y + 1, x + 1] == 0 or no_skip_white:
                mask[:] = 0
                # Fills a connected component with the given color.
                # loDiff – Maximal lower brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # upDiff – Maximal upper brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # flags=4 means that only the four nearest neighbor pixels (those that share an edge) are considered.
                #       8 connectivity value means that the eight nearest neighbor pixels (those that share a corner) will be considered
                rect = cv2.floodFill(
                    im.copy(),
                    mask,
                    (x, y),
                    0,
                    color_difference,
                    color_difference,
                    flags=4 | (255 << 8),
                )
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                segment_size = mask.sum()
                if segment_size > largest_segment:
                    largest_segment = segment_size
                    wallmask = mask[1:-1, 1:-1].copy()
                    # cv2.imshow('rect[2]', mask)
                    # cv2.waitKey(0)

    wallmask = wallmask.astype(np.int64) + ((im.sum(2) == 0).astype(np.int64) * 255)
    wallmask = np.clip(wallmask, 0, 255)
    return wallmask.astype(np.uint8)