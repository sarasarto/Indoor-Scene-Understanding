from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# immagine di riferimento. Vorrei tornare immagini simili a lei

img = cv2.cvtColor(cv2.imread('retrieval_sift/shopping.jpg'), cv2.COLOR_BGR2RGB)
gray_l = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_l,None)
test = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(test)
plt.title("keypoints prima seggiola")
plt.show()

# confronto tutte le immagini nella cartella 
# che simula i dataset di retrievale con shopping.jpg
num_good = []

path = 'retrieval_sift/data_chairs/'
for f in os.listdir(path):

    path2 = os.path.join(path, str(f))
    img2 = cv2.imread(path2, cv2.COLOR_BGR2RGB)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray_2,None)
    test2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(test2)
    #plt.show()

    # cv2.BFMatcher() takes the descriptor of one feature in first set 
    # and is matched with all other features in second set using some distance calculation.
    # And the closest one is returned.

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #print("match trovati: " + str(len(matches)))
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    num_good.append(len(good))
    #print("match buoni trovati: " + str(len(good)))
    img3 = cv2.drawMatchesKnn(img,kp1,img2,kp2, good,None, flags=2)
    plt.imshow(img3),plt.show()

# stampo i punti che soddisfano la percentuale per ogni immagine
for i , img in enumerate(os.listdir(path)):
  print(str(img) + '\t' + str(num_good[i]))
num_good = np.array(num_good)

# decido di tenere solo i tre migliori
num_good_sorted = num_good.argsort()

best = []
for i , img in enumerate(os.listdir(path)):
    if i in num_good_sorted[-3:]:
        best.append(str(img))

print("Le migliori tre corrispondenze: " + str(best))