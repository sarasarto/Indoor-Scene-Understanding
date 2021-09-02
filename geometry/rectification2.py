import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import skimage.segmentation as seg
import skimage.color as color


class GeometryRectification():
    def __init__(self) -> None:
        self.DEBUG = True
    # ------------ CORNER DETECTION ------------

    def findCorners(self, src):
        """
        Find four corners in a given image
        :param src: input image
        :return: corners found
        """
        out_corners = []
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        corners = cv2.goodFeaturesToTrack(gray, 4, 0.009, 5)
        try:
            corners = np.int0(corners)
        except:
            return out_corners, False

        for corner in corners:
            x, y = corner.ravel()
            out_corners.append([x, y])
            cv2.circle(src, (x, y), 3, 255, - 1)

        # check if it's barely regular
        if self.DEBUG:
            cv2.imshow('Corners', src)
            cv2.waitKey(0)

        return out_corners, True




    # ------------ FIND A POLYGON IN THE IMAGE: 2 WAYS ------------

    def rectification_polygon(self, src_img, alpha, beta, threshold):
        """
        Search for polygons to separate the painting from its frame
        :param src_img: ROI image.
        :param alpha: value used to multiply the src_img
                    to change luminosity and contrast.
        :param beta: value used as bias in the src_img
                    to change luminosity and contrast.
        :param threshold: threshold to apply on the grayscale image.
        :return: blank image with the shape of the polygon.
        """
        new_image = np.zeros(src_img.shape, src_img.dtype)
        blank = 255 * np.ones_like(src_img)

        new_image[:, :, :] = np.clip(alpha * src_img[:, :, :] + beta, 0, 255)

        gray_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        _, threshold = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours on the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            cnt_size = w * h
            if area > 10000:
                try:
                    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
                except:
                    return blank, False

                if (len(approx) == 4):
                    cv2.drawContours(blank, [approx], 0, (0, 0, 255), 1)

                    # discard the contour if it's equal to the image
                    if abs(cnt_size - src_img.shape[0] * src_img.shape[1]) <= 0.1:
                        continue
                    return blank, True  # break to the first found

        return blank, False



    def rectification_mask(self, roi_img):
        """
        Segment the image and perform morphological operators to separate the painting from its frame
        :param roi_img: cutted image to rectify.
        :return: mask which delimits the painting found.
        """
        Z = roi_img.reshape((-1, 3))
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        K = 10
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

        # Convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((roi_img.shape))

        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
        lab = cv2.cvtColor(res2, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv2.split(lab)  # split on 3 different channels
        l2 = clahe.apply(l)  # apply CLAHE to the L-channel
        lab = cv2.merge((l2, a, b))  # merge channels
        
        image_slic = seg.slic(lab, n_segments=4, sigma=1, compactness=5, start_label=1)
        image = self.img_as_ubyte(color.label2rgb(image_slic, roi_img, kind='avg', bg_label=0))

        mask = np.zeros(image.shape, dtype=np.uint8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        edges = cv2.Canny(thresh, 100, 200)
        # Perform morpholgical operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find distorted rectangle contour and draw onto a mask
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        rect = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(close, [box], 0, (0, 0, 255), 2)
        cv2.fillPoly(mask, [box], (255, 255, 255))

        mask_seg = np.zeros(image.shape, dtype=np.uint8)
        mask_seg[10:-10, 10:-10] = 255
        mask = mask & mask_seg

        return mask, True


    # ------------ ORDER AND TRANSFORM POINTS ------------

    def order_points(self, pts):
        """
        Given an unordered set of points, order them from upper-left to bottom-right
        :param pts: source points.
        :return: array of ordered points.
        """

        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect


    def four_point_transform(self, image, pts):
        """
        Given an image and 4 points (corners), compute the perspective transform
        and rectify the region delimited by the corners.
        :param image: image to rectify.
        :param pts: corners that delimit the region to rectify.
        :return: rectified image.
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped


    def rectification(self, roi_img):
        """
        Perform the rectification
        :param roi_img: cutted image to rectify
        :return: image rectified if 4 corners are found,
                None otherwise.
        """
        found = False
        i = 0
        params = [[10.0, -500, 60], [1.0, 0, 60], [10.0, -300, 100], [2.0, 0, 120], [7.0, -500, 90],
                [7.0, -500, 200], [1.0, 0, 150], [1.0, 0, 160], [1.0, 0, 120], [7.0, -800, 140]]
        while not found and i != 10:
            mask, found = self.rectification_polygon(roi_img, params[i][0], params[i][1], params[i][2])
            i+=1

        method = 1

        if not found:
            method = 2
            try:
                mask, found = self.rectification_mask(roi_img)
            except:
                found = False

        if found:
            if self.DEBUG:
                print('Method:', method)
            corners_src, found = self.findCorners(mask)
            if found:
                corners_src = np.asarray(corners_src, dtype=np.int)
                result = self.four_point_transform(roi_img, corners_src)
                cv2.imshow('Perspective Transform', result)

                if self.DEBUG:
                    cv2.waitKey(0)
                return result
            else:
                if self.DEBUG:
                    print('No corners found')
        else:
            if self.DEBUG:
                print('No shape found')
        return None