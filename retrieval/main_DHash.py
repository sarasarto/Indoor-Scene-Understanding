import os
from PIL import Image
import matplotlib.pyplot as plt
from image_with_Hash import Images_with_Hash
from DHash import DHash_Helper
import operator

import dhash
# ATTENTION: to use this code please install --> pip install dhash


# prova di funzionamento
if __name__ == '__main__':
  # TO DO: use the retrieval dataset with single objects all together on Drive
  folder_image = '/content/drive/MyDrive/autoenc_data/test/'
  hash_helper = DHash_Helper(folder_image)

  all_hashed = hash_helper.compute_hash_dataset()  
  matched_images = hash_helper.match_furniture(Image.open("../autoencoder_retrieval/sofa.jpg"), all_hashed)

  for i in matched_images:
    print(i.img_name)
    img = Image.open(os.path.join(folder_image, i.img_name))
    plt.imshow(img)
    plt.show()
