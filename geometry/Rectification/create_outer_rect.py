import cv2
import numpy as np


def mask(input, debug=False, **kwargs):
    contours = [ contour for img, contour in input ]
    img = _mask(contours, **kwargs)
    if debug:
        return img, img
    else:
        return img

def _mask(contours, source, pad=0):
    img = source.copy()
    for contour in contours:
        img = draw_rect(contour, img, pad)
    return img

def draw_rect(contour, source, pad=0):
    rect = cv2.boundingRect(contour)
    x,y,w,h = rect
    x -= pad
    y -= pad
    color = [0, 0, 255]
    out = cv2.rectangle(source.copy(), (x, y), (x+w, y+h), color, thickness=3)
    perc = cv2.contourArea(contour) / (w * h)
    cv2.putText(out, f'{perc*100:.01f}%', (x+10, y-10+h), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
    return out

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