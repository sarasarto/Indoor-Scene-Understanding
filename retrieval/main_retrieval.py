from PIL import Image
import matplotlib.pyplot as plt
from retrieval.method_dhash.DHash_utils import DHash_Helper
from retrieval.method_SIFT.SIFT_utils import SIFT_Helper
import cv2 as cv
import numpy as np


# this is just used for test
# in reality we will use the mask
def load_image(box, img):
    image = cv.imread(img)
    x, y, width, height = int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])
    rect = (x, y, width, height)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    #COMMENTO SCRITTO DA BEPPE: QUESTO GRABCUT E' MEGLIO SPOSTARLO IN UNA CLASSE CHE SI OCCUPA SOLO DEL PROCESSING!

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask2[:, :, np.newaxis]
    crop_img = image[y:y + height, x:x + width]
    plt.imshow(crop_img), plt.colorbar(), plt.show()
    return crop_img


if __name__ == '__main__':
    folder_image = 'grabcut_kaggle_dataset'
    test_image = 'test_kaggle'

    action = input('What method do you want to use? [DHash/SIFT/autoencoder]: ')
    data = input('Do you want to use test_kaggle or a single image? [all/single]: ')

    # if you are trying with only one image
    bounding_box = [287, 252, 583, 440]
    my_image = load_image(bounding_box, 'ADE_train_00000278.jpg')

    if action == 'SIFT':
        rs = SIFT_Helper()

        if data == "all":
            # showing for all the test dataset the similar images found

            rs.find_similar_and_print(test_image)
            rs.ask_evaluation(test_image)
        else:
            # if you have only one image

            obj_list, num_good = rs.retrieval(my_image, "bed")
            retr_imgs = rs.print_results(obj_list, num_good, "bed")
            # starting evaluation
            single_AP = rs.get_AP_SIFT(retr_imgs, my_image)
            print(single_AP)

    else:
        if action == "DHash":
            hash_helper = DHash_Helper(folder_image)
            hash_helper = DHash_Helper(folder_image)

            all_hashed = hash_helper.compute_hash_dataset()
            print("all hashed images done!")

            if data == "all":

                # showing for all the test dataset the similar images found
                hash_helper.find_similar_and_print(test_image, all_hashed)

                # asking if you want to do the evaluation
                hash_helper.ask_evaluation(test_image, all_hashed)
            else:
                # only with an image
                matched_images = hash_helper.match_furniture(Image.fromarray(my_image), all_hashed)
                plt.imshow(Image.fromarray(my_image))
                plt.title('Query Image')
                plt.show()

                retr_imgs = hash_helper.print_similar(matched_images, hash_helper.folder)

                # starting evaluation
                single_AP = hash_helper.get_AP_DHash(retr_imgs, my_image)
                print(single_AP)

        else:
            if action == "autoencoder":
                print("still to change")
