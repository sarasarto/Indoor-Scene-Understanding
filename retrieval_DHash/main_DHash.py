import os
from PIL import Image
import matplotlib.pyplot as plt
from image_with_Hash import Images_with_Hash
import operator

import dhash
# ATTENTION: to use this code please install --> pip install dhash


# TO DO: use the retrieval dataset with single objects all together on Drive
folder_image = '/content/drive/MyDrive/autoenc_data/test/'

# computing the hash value for each image in the retrieval dataset
def compute_hash_dataset():
  all_images_hashed = []
  for f in os.listdir(folder_image):
    img = Image.open(os.path.join(folder_image, f))
    row, col = dhash.dhash_row_col(img)
    hashed_image = Images_with_Hash(f, int(dhash.format_hex(row, col), 16))
    all_images_hashed.append(hashed_image)

  return all_images_hashed


# computing the hash bit differences between the input image and the rest of the retrieval dataset
# returning the first 11 most similar objects
def match_furniture(img, all_images_hashed):

  threshold = 40

  img_row, img_col = dhash.dhash_row_col(img)
  img_hash = dhash.format_hex(img_row, img_col)
  img_hash = int(img_hash, 16)
  differences = {}

  # Check difference between img and painting_db
  for idx, single_hash_image in enumerate(all_images_hashed):
    differences[idx] = dhash.get_num_bits_different(img_hash, single_hash_image.hash)

  print(differences)

  sorted_x = sorted(differences.items(), key=operator.itemgetter(1))
  print('ordinati:')
  print(sorted_x)

  # print(sorted_x[:11])
  first_eleven = sorted_x[:11]

  res_img_name = []
  for rel in first_eleven:
    res_img_name.append(all_images_hashed[rel[0]])

  '''res_img_name = []
  for idx , d in enumerate(differences):
    if d <= threshold:
      res_img_name.append(all_images_hashed[idx]) '''

  return res_img_name


if __name__ == '__main__':
  all_hashed = compute_hash_dataset()
  matched_images = match_furniture(Image.open("sofa.jpg"), all_hashed)
  for i in matched_images:
    print(i.img_name)
    img = Image.open(os.path.join(folder_image, i.img_name))
    plt.imshow(img)
    plt.show()
