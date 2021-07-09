from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)


# The most important practical difference is that findContours gives connected contours,
# while Canny just gives edges, which are lines that may or may not be connected to each other. 
def thresh_callback(val):
    threshold = val
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 3)
 
    #cv.imshow('Contours', canny_output)
    #cv.waitKey()
    
    # funzione findsCounter():
    # RETR_EXTERNAL:only retrieve the outermost contour;
    # RETR_TREE (most commonly used): retrieve all contours and reconstruct the entire hierarchy of nested contours;

    # CHAIN_APPROX_NONE: The outline is output in the form of Freeman chain code, and all other methods output polygons (sequence of vertices).
    # CHAIN_APPROX_SIMPLE (most commonly used): Compress the horizontal, vertical and diagonal parts, that is, the function only retains their end parts.
    
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
   
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
   
    # le coordinate dei box sono organizzate in questo modo: (x, y , w , h)
    # (x,y) è il punto in alto a sx
    # w è la width, h è height
    # quindi per valutare l'area basta fare base*altezza*1/2 --> w* h *1/2
    print("coordinate dei box")
    print(boundRect)
       
    area = []
    for b in range(len(boundRect)):
        # Area
        area.append(boundRect[b][2]*boundRect[b][3]*1/2)
    
    print("area ")
    print(area)
    
    # mi interessa solo l'area del rettangolo piu grande,
    # considerando che le nostre foto avranno solo un soggetto unico
    
    max = np.argmax(area)
    
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv.drawContours(drawing, contours_poly, max, color)
    cv.rectangle(drawing, (int(boundRect[max][0]), int(boundRect[max][1])), \
          (int(boundRect[max][0]+boundRect[max][2]), int(boundRect[max][1]+boundRect[max][3])), color, 2)
    
    
    '''
    # qua stampa tutti i contorni che ci sono con i rettangoli
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    
        #cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    '''
    cv.imshow('Contours', drawing)

# lista di immagini di test
images = ['furniture_retrieval/armchair.jpg','furniture_retrieval/tavolo.jpg', 'furniture_retrieval/shopping.jpg', 'furniture_retrieval/lampada.jpg','furniture_retrieval/letto2.jpg',  'furniture_retrieval/table.jpg' , 'furniture_retrieval/sofa.jpg']

for im in images:
    src = cv.imread(im)
    
    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))
    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, src)
 
    max_thresh = 255
    thresh = 100# initial threshold
    
    cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)
    cv.waitKey()