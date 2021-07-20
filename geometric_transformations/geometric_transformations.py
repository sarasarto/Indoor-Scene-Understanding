import cv2
import numpy as np
import math


class GeometryTransformer():
    def __init__(self):
        pass
        
    def order_points(self, pts):
        pts = pts.squeeze()
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
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
        return  height, width
    
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
        k2 = ((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x) / ((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x)
        k3 = ((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y - m1y*m4x) / ((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x)

        # if k2==1 AND k3==1, then the focal length equation is not solvable 
        # but the focal length is not needed to calculate the ratio.
        # I am still trying to figure out under which circumstances k2 and k3 become 1
        # but it seems to be when the rectangle is not distorted by perspective, 
        # i.e. viewed straight on. Then the equation is obvious:
        if k2 == 1 or k3 == 1:
            whRatio = np.sqrt((self.square(m2y-m1y) + self.square(m2x-m1x)) / (self.square(m3y-m1y) + self.square(m3x-m1x)))
        else:
            # f_squared is the focal length of the camera, squared
            # if k2==1 OR k3==1 then this equation is not solvable
            # if the focal length is known, then this equation is not needed
            # in that case assign f_squared= square(focal_length)
            f_squared = -((k3*m3y - m1y)*(k2*m2y - m1y) + (k3*m3x - m1x)*(k2*m2x - m1x)) / ((k3 - 1)*(k2 - 1))
            #The width/height ratio of the original rectangle
            part_1 = (self.square(k2 - 1) + self.square(k2*m2y - m1y)/f_squared + self.square(k2*m2x - m1x)/f_squared)
            part_2 = (self.square(k3 - 1) + self.square(k3*m3y - m1y)/f_squared + self.square(k3*m3x - m1x)/f_squared)
            whRatio = np.sqrt(part_1 / part_2)
            print(part_1, part_2, whRatio)

        # After testing, I found that the above equations 
        # actually give the height/width ratio of the rectangle, 
        # not the width/height ratio. 
        # If someone can find the error that caused this, 
        # I would be most grateful.
        # until then:
        return whRatio
    

    def furniture_rectification(self, img, corners):
        #corners order: tl, bl, br, tr

        corners = corners.astype(np.float32)

        # Here, I have used L2 norm. You can use L1 also.
        width_AD = np.sqrt(((corners[0][0] - corners[-1][0]) ** 2) + ((corners[0][1] - corners[-1][1]) ** 2))
        width_BC = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))
        
        height_AB = np.sqrt(((corners[0][0] - corners[1][0]) ** 2) + ((corners[0][1] - corners[1][1]) ** 2))
        height_CD = np.sqrt(((corners[2][0] - corners[-1][0]) ** 2) + ((corners[2][1] - corners[-1][1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))

        '''
        dst_pts = np.array([[0,0], #top
                            [width-1,0], #bottom
                            [height-1,width-1],  #left
                            [0,height-1]], dtype='float32') #right

         
        '''          
    
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])


        M = cv2.getPerspectiveTransform(corners, output_pts)

        warped_img = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
        print(warped_img.shape)
        return warped_img

'''
segmentation = cv2.imread(masks_path)
instances = np.unique(segmentation[:,:,0])
instances = instances[1:]
print(instances)
masks = segmentation[:,:,0] == instances[:,None,None]

#considero per il momento solo l'istanza del letto
#che ha indice 11
pos = np.where(masks[8])
xmin = np.min(pos[1])
xmax = np.max(pos[1])
ymin = np.min(pos[0])
ymax = np.max(pos[0])

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
roi = masks[8][ymin:ymax, xmin:xmax]
img = img[ymin:ymax, xmin:xmax]
roi = np.where(roi==0, 0, 255)
roi = np.float32(roi)
img[roi==0] = 255


rotated = ndimage.rotate(img, 45)
print(rotated)
cv2.imshow('immagine ruotata', rotated)
cv2.waitKey()


sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img,None)



keypoints_matrix = np.zeros((len(kp1),2))
for i, keypoint in enumerate(kp1):
    keypoints_matrix[i, 0] = keypoint.pt[0]
    keypoints_matrix[i, 1] = keypoint.pt[1]


test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(test1)
plt.show()


tl = (np.min(keypoints_matrix[:,0]), np.max(keypoints_matrix[:,1]))
tr = (np.max(keypoints_matrix[:,0]), np.max(keypoints_matrix[:,1]))
br = (np.max(keypoints_matrix[:,0]), np.min(keypoints_matrix[:,1]))
bl = (np.min(keypoints_matrix[:,0]), np.min(keypoints_matrix[:,1]))

x = [tl[0], tr[0], br[0], bl[0]]
#print(x)
y = [tl[1], tr[1], br[1], bl[1]]
print(y)

#test1 = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#print(kp1[0].pt)
implot = plt.imshow(img)
plt.scatter(x,y, c='r')
plt.title("keypoints del letto")
plt.show()
'''