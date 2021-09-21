import cv2
import numpy as np

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


def _mask_from_contour(img, contour):
    """
        Applies the proper median filter for smoothing the frame's sides
        Parameters
        ----------
        img : the input image
        Returns
        -------
        img : the smoothed image
    """
    canvas = np.zeros_like(img)
    cv2.fillPoly(canvas, pts=[contour], color=(255, 255, 255))
    return canvas


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

    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32)
    labels, centers = cv2.kmeans(pts, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1:]
    labels = labels.reshape(-1)

    groups = [[] for _ in range(k)]
    for label, line in zip(labels, lines):
        groups[label].append(line)

    return groups


def find_all_intersections(groups):
    """
    Find all possible points of intersection between all lines of all groups
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
    #Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (default_criteria_type, 10, 1.0)
    pts = np.array([[point[0], point[1]] for point in points], dtype=np.float32)
    labels, centers = cv2.kmeans(pts, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[1:]

    corners = []
    for center in centers:
        x = int(np.round(center[0]))
        y = int(np.round(center[1]))
        corners.append([x, y])
    return corners


def _find_corners(lines):
    """
    Given a list of lines it finds the 4 corners of the object
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