import dhash
import operator
import os
from PIL import Image
import cv2
from retrieval.method_dhash.image_with_Hash import Images_with_Hash


class DHashHelper():
    def __init__(self, folder_grabcut='retrieval/grabcut_kaggle_dataset_folder', folder='retrieval/kaggle_dataset_folder_jpg'):
        self.folder = folder
        self.folder_grabcut = folder_grabcut

    # return: the hash value for each image in the correct label of the retrieval dataset
    def compute_hash_dataset(self, label):
        path = self.folder_grabcut + '/' + label
        hashed_images = []
        for f in os.listdir(path):
            img = Image.open(os.path.join(path, f))
            row, col = dhash.dhash_row_col(img)
            image = Images_with_Hash(f, int(dhash.format_hex(row, col), 16))
            hashed_images.append(image)

        return hashed_images

    # computing the hash bit differences between the input image and the rest of the retrieval dataset
    # return: the first 5 most similar objects, not considering a threshold value
    def retrieval(self, query_img, label):
        #hashing only images with that label
        hashed_images = self.compute_hash_dataset(label)

        img_row, img_col = dhash.dhash_row_col(query_img)
        img_hash = dhash.format_hex(img_row, img_col)
        img_hash = int(img_hash, 16)
        differences = {}

        # Computing Hamming distance between query and dataset images
        for idx, single_hash_image in enumerate(hashed_images):
            differences[idx] = dhash.get_num_bits_different(img_hash, single_hash_image.hash)

        # sorting images of dataset by difference value
        sorted_x = sorted(differences.items(), key=operator.itemgetter(1))

        # taking only the first 5 images
        first_five = sorted_x[:5]

        img_names = []
        for rel in first_five:
            img_names.append(hashed_images[rel[0]])

        '''res_img_name = []
        for idx , d in enumerate(differences):
            if d <= threshold:
            res_img_name.append(all_images_hashed[idx]) '''

        #returnig 5 most similar images
        similar_images = []
        for (j, i) in enumerate(img_names):
            m_img = cv2.cvtColor(cv2.imread(os.path.join(self.folder, label, i.img_name)), cv2.COLOR_BGR2RGB)
            similar_images.append(m_img)
        return similar_images
