import cv2
import numpy as np

def color_contours(img, contours):
    canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # canvas = np.zeros_like(img)
    for contour in contours:
        color = np.random.randint(256, size=3).tolist()
        cv2.fillPoly(canvas, pts=[contour], color=color)
    return canvas

def find_contours(input: np.array, debug=False):
    """
    Dilates an image by using a specific structuring element
    The function retrieves contours from the binary image using the algorithm [Suzuki85].
    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#dilate
    ----------
    img : np.array
        image where to apply the dilatation
    """
    # CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours.
    # CV_CHAIN_APPROX_NONE stores absolutely all the contour points.
    # That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors,
    # that is, max(abs(x1-x2),abs(y2-y1))==1.
    img = input
    contours = _find_contours(img)
    if debug:
        canvas = color_contours(img.copy(), contours)
        return (img, contours), canvas
    else:
        return (img, contours)

def _find_contours(img: np.array):
    """
    Dilates an image by using a specific structuring element
    The function retrieves contours from the binary image using the algorithm [Suzuki85].
    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#dilate
    ----------
    img : np.array
        image where to apply the dilatation
    """
    # CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours.
    # CV_CHAIN_APPROX_NONE stores absolutely all the contour points.
    # That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors,
    # that is, max(abs(x1-x2),abs(y2-y1))==1.
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) # cv2.CHAIN_APPROX_SIMPLE to save memory
    return contours

def couldBePainting(img: np.array, bounder, contour, width, height, area_percentage):
    bounder_width, bounder_height = bounder[2], bounder[3]
    bounder_area = bounder_width * bounder_height
    # Check that the rect is smaller than the entire image and bigger than a certain size
    if bounder_area < img.shape[0]*img.shape[1]*0.9 and bounder_width > width and bounder_height > height:
        # Extra to remove floors when programming
        if cv2.contourArea(contour) > bounder_area * area_percentage:
            return True
    return False


def find_possible_contours(input, min_width=50, min_height=50, min_area_percentage=.6, debug=False):
    img, contours = input
    painting_contours = _find_possible_contours(img, contours, min_width, min_height, min_area_percentage)
    result = [(img, contour) for contour in painting_contours]
    if debug:
        canvas = color_contours(img.copy(), painting_contours)
        return result, canvas
    else:
        return result

def _find_possible_contours(img, contours, min_width=50, min_height=50, min_area_percentage=.6):
    painting_contours = []
    for contour in contours:
        bounder = cv2.boundingRect(contour)
        if couldBePainting(img, bounder, contour, min_width, min_height, min_area_percentage):
            painting_contours.append(contour)
    return painting_contours