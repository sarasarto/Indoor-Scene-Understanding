import dhash
import operator
import os
from PIL import Image
from retrieval.image_with_Hash import Images_with_Hash
from retrieval.evaluation import RetrievalMeasure

class DHash_Helper():
    def __init__(self, folder):
        self.folder = folder
        self.rm = RetrievalMeasure()
    # computing the hash value for each image in the retrieval dataset
    def compute_hash_dataset(self):
        all_images_hashed = []
        for f in os.listdir(self.folder):
            img = Image.open(os.path.join(self.folder, f))
            row, col = dhash.dhash_row_col(img)
            hashed_image = Images_with_Hash(f, int(dhash.format_hex(row, col), 16))
            all_images_hashed.append(hashed_image)

        return all_images_hashed


    # computing the hash bit differences between the input image and the rest of the retrieval dataset
    # returning the first 11 most similar objects, not considering a threshold value
    def match_furniture(self , img, all_images_hashed):

        threshold = 40

        img_row, img_col = dhash.dhash_row_col(img)
        img_hash = dhash.format_hex(img_row, img_col)
        img_hash = int(img_hash, 16)
        differences = {}

        # Computing Hamming distance bewtween query and dataset image 
        for idx, single_hash_image in enumerate(all_images_hashed):
            differences[idx] = dhash.get_num_bits_different(img_hash, single_hash_image.hash)

        #print(differences)

        # sorting images of dataset by difference value
        sorted_x = sorted(differences.items(), key=operator.itemgetter(1))
        print('ordinati:')
        print(sorted_x)

        # taking only the first 5 images
        first_eleven = sorted_x[:6]

        res_img_name = []
        for rel in first_eleven:
            res_img_name.append(all_images_hashed[rel[0]])

        '''res_img_name = []
        for idx , d in enumerate(differences):
            if d <= threshold:
            res_img_name.append(all_images_hashed[idx]) '''

        return res_img_name

    def get_AP_DHash(self, retrieved_imgs, img_ref):

        user_resp = self.rm.get_user_relevance_autoencoder(img_ref, retrieved_imgs)
        print("user responses:")
        print(user_resp)
        AP = self.rm.get_AP(user_resp, 5)
        print("Average Precision: " + str(AP))

        return AP

    def compute_MAP_DHash(self, AP_vector):
        return self.rm.compute_MAP(AP_vector)

