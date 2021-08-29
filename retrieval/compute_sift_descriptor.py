import json
import os
def main():

    data = json.load(open('Annotations_Kaggle.json'))
    dataset_sift_descriptor = []
    for annotated_img in data:
        img_description = {}

        img_description['name'] = annotated_img['image']
        img_description['label'] = annotated_img["annotations"][0]["label"]
        
        path = 

            # ho aggiunto questo path
            path = self.grabcut_path + '/' + label
            path = os.path.join(path, annotated_img["image"])

            if os.path.isfile(path):
                obj_list.append(annotated_img["image"])

                img2 = cv.imread(path, cv.COLOR_BGR2RGB)
                gray_2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

                sift = cv.SIFT_create()

                kp2, des2 = sift.detectAndCompute(gray_2, None)
                kp.append(kp2)
                des.append(des2)


if __name__ == '__main__':
    main()