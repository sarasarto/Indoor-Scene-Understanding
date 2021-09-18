import cv2
import numpy as np
from image_viewer import ImageViewer
import math
from pre_processing import _mean_shift_segmentation
from background_detection import _mask_largest_segment
from cleaning import _closing, _invert, _add_padding
from components_selection import _find_contours
from components_selection import _find_possible_contours
from contour_pre_processing import  _clean_frames_noise, _mask_from_contour
from contour_pre_processing import _apply_median_filter
from contour_pre_processing import _apply_edge_detection
from corners_detection import _hough
from corners_detection import _find_corners
from create_outer_rect import rect_contour


class image_rectification:

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
            '''if not contours:
            # convert img to grey
            img_grey = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
            # set a thresh
            thresh = 30
            # get threshold image
            ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
            # find contours
            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)'''

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

        rgbImage_num = rgbImage.copy()
        # rgbImage = cv2.rotate(rgbImage, cv2.ROTATE_90_CLOCKWISE)
        for i, corners in enumerate(object_contours):
            rgbImage_num = cv2.putText(rgbImage_num, str(i + 1), self.mean_center(corners),
                                       cv2.FONT_HERSHEY_SIMPLEX, 5,
                                       (0, 0, 255), 20, cv2.LINE_AA)
        for i, corners in enumerate(object_contours):
            iv = ImageViewer()
            iv.add(rgbImage_num, 'original', cmap='bgr')
            sec = self.cut_section(rgbImage_num, corners)
            iv.add(sec, 'target', cmap='bgr')
            cv2.imwrite('data_test/target.jpg', rgbImage)
            img = self.four_point_transform(rgbImage, np.array(corners))
            if not img is None:
                iv.add(img, 'VIT painting {}'.format(i + 1), cmap='bgr')
                cv2.imwrite('data_test/vit.jpg', img)
            img = self.painting_rectification(rgbImage, np.array(corners))
            if not img is None:
                iv.add(img, 'STACK painting {}'.format(i + 1), cmap='bgr')
                cv2.imwrite('data_test/stack.jpg', img)

            iv.show()

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


    def painting_rectification(self, img, corners):
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


    def perspective_dim(self, tl, tr, br, bl):
        width_B = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_T = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        height_R = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_L = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        if 0.0 in [width_B, width_T, height_R, height_L]:
            return None, None

        height_ratio = max(height_L / height_R, height_R / height_L) - 1
        width_ratio = max(width_B / width_T, width_T / width_B) - 1

        if 0.0 in [height_ratio, width_ratio]:
            return None, None
        max_width = max(width_B, width_T)
        max_height = max(height_L, height_R)
        width = np.sqrt(max_width ** 2 * (1 + height_ratio ** 2))
        height = np.sqrt(max_height ** 2 * (1 + width_ratio ** 2))

        return int(height), int(width)


    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        # rect = pts
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width, height = self.perspective_dim(tl, tr, br, bl)
        if width is None or height is None:
            return None
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [height - 1, 0],
            [height - 1, width - 1],
            [0, width - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (height, width))
        # return the warped image
        return warped


    def remove_pad(self, pts, pad):
        return [[x - pad, y - pad] for x, y in pts]


    def cut_section(self, img, corners):
        x_max = max([x[0][0] for x in corners])
        x_min = min([x[0][0] for x in corners])
        y_max = max([x[0][1] for x in corners])
        y_min = min([x[0][1] for x in corners])
        y_min, x_min = max(y_min, 1), max(x_min, 1)
        return img[y_min:y_max, x_min:x_max]


    def mean_center(self, pts):
        x, y = 0, 0
        for pt in pts:
            x += pt[0, 0]
            y += pt[0, 1]
        x = x  # len(pts)
        y = y  # len(pts)
        return (x, y)

if __name__ == "__main__":

    # codice per far funzionare Imageviewer su MacOS
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    # Put figure window on top of all other windows
    fig.canvas.manager.window.attributes('-topmost', 1)
    # After placing figure window on top, allow other windows to be on top of it later
    fig.canvas.manager.window.attributes('-topmost', 0)

    # valori per prova contorni porta
    # corners = [[30, 25], [347, 9], [37, 578],
    #           [348,629]]
    # valori per prova contorni quadro
    #corners = [[100, 400], [1600, 40], [540, 2160],
    #           [1890, 1880]]
    # valori per prova contorni finestra
    # corners = [[11, 16], [261,58], [17, 194],
    #           [256,179]]
    # corners = [[0,0], [0, rgbImage.shape[1]-1], [rgbImage.shape[0]-1, 0], [rgbImage.shape[1]-1, rgbImage.shape[0]-1]]

    img_rect = image_rectification()

    TEST_IMAGE = ['test_image/finestre-pvc-1.jpg',
                  'test_image/edc090117catroux12-1604942537.jpg',
                  'test_image/photo-1595526114035-0d45ed16cfbf.jpeg',
                  'test_image/prova1.jpeg',
                  'test_image/prova2.jpeg',
                  'test_image/10603 copia.jpeg']

    # rgbImage = cv2.imread(PERSPECTIVE)
    for filename in TEST_IMAGE[::-1]:
        rgbImage = cv2.imread(filename)
        img_rect.rectify(rgbImage)