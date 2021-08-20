import os
from PIL import Image
import matplotlib.pyplot as plt
from retrieval.DHash_utils import DHash_Helper

import dhash

# ATTENTION: to use this code please install --> pip install dhash


if __name__ == '__main__':
    # TO DO: use the retrieval dataset with single objects all together on Drive
    # using the one with grabcut applied
    folder_image = 'grabcut_kaggle_dataset'
    test_image = 'test_kaggle'
    hash_helper = DHash_Helper(folder_image)

    all_hashed = hash_helper.compute_hash_dataset()
    print("done")
    AP_test = []
    for img in os.listdir(test_image):
        path = test_image + '/' + img
        matched_images = hash_helper.match_furniture(Image.open(path), all_hashed)
        plt.imshow(Image.open(path))
        plt.title('Query Image')
        plt.show()

        retr_imgs = hash_helper.print_similar(matched_images, folder_image)

        # starting evaluation
        single_AP = hash_helper.get_AP_DHash(retr_imgs, img)
        AP_test.append(single_AP)
        print("AP vector")
        print(AP_test)

    # AP_test = [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8333333333333333, 0.95, 1.0, 1.0, 0.7555555555555555, 1.0, 0.8333333333333333, 1.0, 1.0, 1.0, 1.0, 1.0]
    print("computing MAP on test kaggle dataset ")
    print(hash_helper.compute_MAP_DHash(AP_test))
