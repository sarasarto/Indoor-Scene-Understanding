import cv2
import numpy as np


def _mask_largest_segment(im: np.array, color_difference, scale_percent=1.0, x_samples=2, no_skip_white=False):
    """
    ----------
    img : np.array
        image where to find the largest element
    color_difference : int
        The distance from colors to permit.
    x_samples : int
        number of samples that will be tested horizontally in the image
    """
    im = im.copy()

    h = im.shape[0]
    w = im.shape[1]
    color_difference = (color_difference,) * 3

    stride = int(w / x_samples)
    if w == 0:
        return None
    mask = np.zeros((im.shape[0] + 2, im.shape[1] + 2), dtype=np.uint8)
    background_mask = mask[1:-1, 1:-1].copy()
    largest_segment = 0
    for y in range(0, im.shape[0], stride):
        for x in range(0, im.shape[1], stride):
            if mask[y + 1, x + 1] == 0 or no_skip_white:
                mask[:] = 0

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
                    background_mask = mask[1:-1, 1:-1].copy()

    background_mask = background_mask.astype(np.int64) + ((im.sum(2) == 0).astype(np.int64) * 255)
    background_mask = np.clip(background_mask, 0, 255)
    return background_mask.astype(np.uint8)