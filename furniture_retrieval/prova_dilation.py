import cv2

def edge_detection(val):
    threshold = val
    image = 'furniture_retrieval/armchair.jpg'
    image = cv2.imread(image)

    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3,3))
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(image, kernel)
    _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIM)


max_thresh = 255
thresh = 100# initial threshold
image = 'furniture_retrieval/armchair.jpg'
image = cv2.imread(image)
cv2.grabCut(image)

source_window = 'Source'
cv2.imshow(source_window, image)
cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, edge_detection)
edge_detection(thresh)


cv2.waitKey()