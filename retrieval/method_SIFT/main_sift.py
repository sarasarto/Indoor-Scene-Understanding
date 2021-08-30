import cv2 as cv
from matplotlib import pyplot as plt
import os
from PIL import Image
from retrieval.SIFT_utils import SIFT_Helper

if __name__ == '__main__':

    folder_image = 'grabcut_kaggle_dataset'
    test_image = 'test_kaggle'

    rs = SIFT_Helper()
    # testing with test_kaggle that already have grabcut
    # when using with mask nn we need these two next steps
    # bounding_box = [287, 252, 583, 440]
    # my_image = rs.load_image(bounding_box, 'ADE_train_00000278.jpg')

    for img in os.listdir(test_image):
        path = test_image + '/' + img
        plt.imshow(Image.open(path))
        plt.title('Query Image')
        plt.show()

        label = input('Write the class label [lamp, sofa, armchair, chair, bed, bicycle]: ')
        my_image = cv.imread(path)
        # non faccio controlli che sia scritto bene, è solo una prova
        # avrei potuto cercare immagine nell annotation di kaggle ma tanto poi useremo la mask


        obj_list, num_good = rs.retrieval(my_image, label)
        rs.print_results(obj_list, num_good, label)


        # ATTENZIONE! ALCUNE IMMAGINI DI TEST_KAGGLE NON FUNZIONANO CON SIFT!!!!
        # CONTROLLARE IL PERCHE'