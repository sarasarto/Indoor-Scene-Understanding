import numpy as np
import cv2


class RetrievalHelper():
    # applying grabcut on the image
    def extract_query_foreground(self, image, mask):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]

        return image