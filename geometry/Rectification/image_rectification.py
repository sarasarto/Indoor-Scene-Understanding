import cv2
import numpy as np
import math
from geometry.Rectification.pre_processing import _mean_shift_segmentation
from geometry.Rectification.background_detection import _mask_largest_segment
from geometry.Rectification.cleaning import _closing, _invert, _add_padding
from geometry.Rectification.components_selection import _find_contours
from geometry.Rectification.components_selection import _find_possible_contours
from geometry.Rectification.contour_pre_processing import  _clean_frames_noise, _mask_from_contour
from geometry.Rectification.contour_pre_processing import _apply_median_filter
from geometry.Rectification.contour_pre_processing import _apply_edge_detection
from geometry.Rectification.corners_detection import _hough
from geometry.Rectification.corners_detection import _find_corners
from geometry.Rectification.create_outer_rect import rect_contour



class ImageRectifier:

    def rectify(self, rgbImage):
        color_diff = 1
        contours = []
        while not contours:
            if color_diff == 25:
                break
            out = _mean_shift_segmentation(rgbImage)
            out = _mask_largest_segment(out, color_diff)
            out = _closing(out)
            out = _invert(out)
            out = _add_padding(out, 1)
            contours = _find_contours(out)
            contours = _find_possible_contours(out, contours)
            color_diff = color_diff+1

        object_contours = []
        max_contour = 0
        for contour in contours:
            if cv2.contourArea(contour) < max_contour:
                continue
            if len(object_contours) > 0:
                object_contours.pop()
            max_contour = cv2.contourArea(contour)
            _, _, w, h = cv2.boundingRect(contour)
            found_correct_shape = False
            for_out = _mask_from_contour(out, contour)
            for_out = _clean_frames_noise(for_out)
            for_out = _apply_median_filter(for_out)
            for_out = _apply_edge_detection(for_out)
            lines = _hough(for_out)
            if lines is not None:
                corners = _find_corners(lines)
                if corners is not None:
                    pts = np.array(corners, np.int32)
                    pts = cv2.convexHull(pts)
                    pts = pts.reshape((-1, 1, 2))
                    pts_ratio = cv2.contourArea(contour) / (cv2.contourArea(pts) + 1)
                    if pts_ratio < 1.2 and pts_ratio > cv2.contourArea(contour) / (w * h):
                        object_contours.append(pts)
                        found_correct_shape = True
                if not found_correct_shape:
                    epsilon = 0.1 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                    if len(approx) == 4:
                        object_contours.append(
                            approx)  # UnboundLocalError: local variable 'pts' referenced before assignment
                        found_correct_shape = True

                if not found_correct_shape:
                    object_contours.append(rect_contour(contour, 1))

        if not object_contours:
            print("It isn't possible to find any object to rectify")
            img = []
            return img

        corners = object_contours[0]

        img = self.object_rectification(rgbImage, np.array(corners))

        return img


    def square(self, x):
        return x * x


    def aspect_ratio(self, corners, img):
        tl, tr, br, bl = self.order_points(corners)
        m1x, m1y = bl
        m2x, m2y = br
        m3x, m3y = tl
        m4x, m4y = tr
        v0 = img.shape[0] / 2
        u0 = img.shape[1] / 2
        # in case it matters: licensed under GPLv2 or later
        # legend:
        # square(x)  = x*x
        # sqrt(x) = square root of x

        # let m1x,m1y ... m4x,m4y be the (x,y) pixel coordinates
        # of the 4 corners of the detected quadrangle
        # i.e. (m1x, m1y) are the cordinates of the first corner,
        # (m2x, m2y) of the second corner and so on.
        # let u0, v0 be the pixel coordinates of the principal point of the image
        # for a normal camera this will be the center of the image,
        # i.e. u0=IMAGEWIDTH/2; v0 =IMAGEHEIGHT/2
        # This assumption does not hold if the image has been cropped asymmetrically

        # first, transform the image so the principal point is at (0,0)
        # this makes the following equations much easier
        m1x -= u0
        m1y -= v0
        m2x -= u0
        m2y -= v0
        m3x -= u0
        m3y -= v0
        m4x -= u0
        m4y -= v0

        # temporary variables k2, k3
        k2 = ((m1y - m4y) * m3x - (m1x - m4x) * m3y + m1x * m4y - m1y * m4x) / (
                    (m2y - m4y) * m3x - (m2x - m4x) * m3y + m2x * m4y - m2y * m4x)
        k3 = ((m1y - m4y) * m2x - (m1x - m4x) * m2y + m1x * m4y - m1y * m4x) / (
                    (m3y - m4y) * m2x - (m3x - m4x) * m2y + m3x * m4y - m3y * m4x)

        # if k2==1 AND k3==1, then the focal length equation is not solvable
        # but the focal length is not needed to calculate the ratio.
        # I am still trying to figure out under which circumstances k2 and k3 become 1
        # but it seems to be when the rectangle is not distorted by perspective,
        # i.e. viewed straight on. Then the equation is obvious:
        if k2 == 1 or k3 == 1:
            whRatio = np.sqrt((self.square(m2y - m1y) + self.square(m2x - m1x)) / (self.square(m3y - m1y) + self.square(m3x - m1x)))
        else:
            # f_squared is the focal length of the camera, squared
            # if k2==1 OR k3==1 then this equation is not solvable
            # if the focal length is known, then this equation is not needed
            # in that case assign f_squared= square(focal_length)
            f_squared = -((k3 * m3y - m1y) * (k2 * m2y - m1y) + (k3 * m3x - m1x) * (k2 * m2x - m1x)) / ((k3 - 1) * (k2 - 1))
            # The width/height ratio of the original rectangle
            part_1 = (self.square(k2 - 1) + self.square(k2 * m2y - m1y) / f_squared + self.square(k2 * m2x - m1x) / f_squared)
            part_2 = (self.square(k3 - 1) + self.square(k3 * m3y - m1y) / f_squared + self.square(k3 * m3x - m1x) / f_squared)
            whRatio = np.sqrt(np.abs(part_1) / np.abs(part_2))
            print(part_1, part_2, whRatio)

        # After testing, I found that the above equations
        # actually give the height/width ratio of the rectangle,
        # not the width/height ratio.
        # If someone can find the error that caused this,
        # I would be most grateful.
        # until then:
        return whRatio


    def object_rectification(self, img, corners):
        rect = self.order_points(corners)
        tl, tr, br, bl = rect
        whRatio = self.aspect_ratio(corners, img)
        if math.isnan(whRatio):
            return None
        width, height = self.perspective_dim_ratio(whRatio, tl, tr, br, bl)
        if width is None or height is None:
            return None
        dst = np.array([
            [0, 0],
            [height - 1, 0],
            [height - 1, width - 1],
            [0, width - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (height, width))
        # return the warped image
        return warped


    def order_points(self, pts):
        pts = pts.squeeze()
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect


    def perspective_dim_ratio(self, whRatio, tl, tr, br, bl):
        height_R = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_L = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        height = max(height_L, height_R)
        width = height * whRatio
        try:
            height, width = int(height), int(width)
        except:
            height, width = None, None
        return height, width
