import dhash
import operator
import os
from PIL import Image
from retrieval.image_with_Hash import Images_with_Hash
from retrieval.evaluation import RetrievalMeasure
import matplotlib.pyplot as plt


class DHashHelper():
    def __init__(self, folder='grabcut_kaggle_dataset_folder'):
        self.folder = folder
        self.rm = RetrievalMeasure()

    # computing the hash value for each image in the retrieval dataset
    def compute_hash_dataset(self, label):
        path = self.folder + '/' + label
        hashed_images = []
        for f in os.listdir(path):
            img = Image.open(os.path.join(path, f))
            row, col = dhash.dhash_row_col(img)
            image = Images_with_Hash(f, int(dhash.format_hex(row, col), 16))
            hashed_images.append(image)

        return hashed_images

    # computing the hash bit differences between the input image and the rest of the retrieval dataset
    # returning the first 11 most similar objects, not considering a threshold value
    def retrieval(self, query_img, label):
        #hashing only images with that label
        hashed_images = self.compute_hash_dataset(label)

        threshold = 40

        img_row, img_col = dhash.dhash_row_col(query_img)
        img_hash = dhash.format_hex(img_row, img_col)
        img_hash = int(img_hash, 16)
        differences = {}

        # Computing Hamming distance between query and dataset image 
        for idx, single_hash_image in enumerate(hashed_images):
            differences[idx] = dhash.get_num_bits_different(img_hash, single_hash_image.hash)


        # sorting images of dataset by difference value
        sorted_x = sorted(differences.items(), key=operator.itemgetter(1))

        # taking only the first 5 images
        first_eleven = sorted_x[:6]

        res_img_name = []
        for rel in first_eleven:
            res_img_name.append(hashed_images[rel[0]])

        return res_img_name

    def print_results(self, matched_images, folder_image):
        retr_imgs = []
        for (j, i) in enumerate(matched_images):
            m_img = Image.open(os.path.join(folder_image, i.img_name))
            retr_imgs.append(m_img)

            plt.figure(1, figsize=(20, 10))
            plt.subplot(1, 6, j + 1)
            plt.imshow(m_img)
            plt.suptitle("5 most similar images")

        plt.show()
        print("retri image")
        print(retr_imgs)
        return retr_imgs
