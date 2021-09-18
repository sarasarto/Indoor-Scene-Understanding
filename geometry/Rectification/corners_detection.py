import cv2
import numpy as np
import math

def draw_lines(img, lines, pad):
    canvas = np.stack((img,)*3, axis=-1)
    color = np.random.randint(256, size=3).tolist()
    if not lines is None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)) - pad, int(y0 + 1000*(a)) - pad)
            pt2 = (int(x0 - 1000*(-b)) - pad, int(y0 - 1000*(a)) - pad)
            cv2.line(canvas, pt1, pt2, color, 3, cv2.LINE_AA)
    return canvas

def _hough(img):
    """
    Return the lines found in the image
    Parameters
    ----------
    img : np.array
        image in grayscale or black and white form
    Returns
    -------
    list
        list of all lines found in the image, None if no image is found
    """
    lines = cv2.HoughLines(img, 1, np.pi / 180, 40, None, 0, 0)
    return lines

def hough(input, pad=0, debug=False):
    lines = _hough(input)
    if debug:
        canvas = draw_lines(input, lines, pad)
        return (input, lines), canvas
    else:
        return (input, lines)


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    try:
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]
    except np.linalg.LinAlgError:
        return None


def groups_by_angle(lines, k=2):
    """
    Divide in cluster the lines based on its angle
    Parameters
    ----------
    lines : list
        list of all lines found in the image
    k : int
        number of clusters
    Returns
    -------
    list
        returns a list of k list, where k is the number of clusters. Each line is in the list
        of its cluster
    """
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (default_criteria_type, 10, 1.0)

    angles = np.array([line[0][1] for line in lines])
    """
    Why 2*angle? (angle is in range [0, pi])
    Normally:   
        if line is vertical (pi/2)          sin=0   cos=1
        if line is oblique (pi/4)           sin=0.5 cos=0.5
        if line is oblique (3/4 pi)         sin=0.5 cos=-0.5
        if line is horizontal (0 or pi)     sin=1   cos=0
    Multiplied by 2:
        if line is vertical (pi/2)          sin=0 cos=-1
        if line is oblique (pi/4)           sin=1 cos=0
        if line is oblique (3/4 pi)         sin=-1 cos=0
        if line is horizontal (0 or pi)     sin=0 cos=1   
    Actually I think that the 2* multiplication is not so crucial,
    but in the script that I used as "inspiration" use it,
    so I did the same. 
    """
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32)
    labels, centers = cv2.kmeans(pts, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1:]
    labels = labels.reshape(-1)

    groups = [[] for _ in range(k)]
    for label, line in zip(labels, lines):
        groups[label].append(line)

    return groups


def find_all_intersections(groups):
    """
    Find all possible points of interection between all lines of all groups
    Parameters
    ----------
    groups : list
        list of groups of lines
    Returns
    -------
    list
        returns a list of points [x, y]
    """
    points = []
    for index, group in enumerate(groups[:-1]):
        for line in group:
            for group2 in groups[index + 1:]:
                for line2 in group2:
                    point = intersection(line, line2)
                    if not point is None:
                        points.append(point)
    return points


def find_four_corners(points):
    """
    Find 4 possible points corners using kmeans
    Parameters
    ----------
    points : list
        list of points [x, y]
    Returns
    -------
    list
        returns a list of the corners points [x, y]
    """
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (default_criteria_type, 10, 1.0)
    pts = np.array([[point[0], point[1]] for point in points], dtype=np.float32)
    labels, centers = cv2.kmeans(pts, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1:]

    corners = []
    for center in centers:
        x = int(np.round(center[0]))
        y = int(np.round(center[1]))
        corners.append([x, y])
    return corners


def draw_corners(img, corners):
    if len(img.shape) < 3:
        canvas = np.stack((img,) * 3, axis=-1)
    else:
        canvas = img.copy()
    color = np.random.randint(256, size=3).tolist()
    for point in corners:
        cv2.circle(canvas, (point[0], point[1]), 20, color, -1)
    return canvas


def find_corners(input, debug=False):
    img, lines = input
    corners = _find_corners(lines)
    if corners is None:
        return None if not debug else None, None
    if debug:
        canvas = draw_corners(img, corners)
        return corners, canvas
    else:
        return corners


def _find_corners(lines):
    """
    Given a list of lines it finds the 4 corners of the painting
    Parameters
    ----------
    lines : list
        list of lines
    Returns
    -------
    list
        returns a list of the corners points [x, y]
    """
    if lines is None:
        return None
    groups = groups_by_angle(lines)
    points = find_all_intersections(groups)
    if len(points) < 4:
        return None
    corners = find_four_corners(points)
    return corners