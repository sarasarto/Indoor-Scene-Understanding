import cv2
import numpy as np


def rect_contour(contour, pad=0):
    rect = cv2.boundingRect(contour)
    x,y,w,h = rect
    x -= pad
    y -= pad
    return rect_to_contour(x,y,w,h)


def rect_to_contour(x,y,w,h):
    pts = [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
        ]
    pts = np.array(pts, np.int32)
    pts = cv2.convexHull(pts)
    pts = pts.reshape((-1,1,2))
    return pts