from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.cvtColor(cv2.imread('retrieval_sift/divano_simile.jpg'), cv2.COLOR_BGR2RGB)
gray_l = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_l,None)
test = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(test)
plt.title("keypoints divano")
plt.show()

# confronto tutte le immagini in images con shopping.jpg
images = ['retrieval_sift/divano_simile.jpg' ,'retrieval_sift/divano_rosso.jpg' ,  'retrieval_sift/divano_verde.jpg']
num_good = []
for test in images:
    print(test)
    img2 = cv2.imread(test, cv2.COLOR_BGR2RGB)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
  
    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray_2,None)
    test2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(test2)
    plt.title('keypoints ' + str(test))
    plt.show()
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    good = []
    perc = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            perc.append(m.distance)
            good.append([m])
    num_good.append(len(good))


for i, img in enumerate(images):
  print(str(img) + '\t' + str(num_good[i]))

num_good = np.array(num_good)

print('\n\n')
for i in num_good.argsort()[:3]:
  print(images[i])