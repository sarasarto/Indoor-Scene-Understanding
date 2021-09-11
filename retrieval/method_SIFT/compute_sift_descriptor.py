import json
import pickle
import cv2

def main():
    dataset_path = 'retrieval/grabcut_dataset_folder'
    descriptors_path = 'retrieval/method_SIFT/descriptors.pkl'

    data = json.load(open('retrieval/Annotations_Kaggle.json'))

    dataset_keypoints = [] #list of list. each list contains different number of keypoints
    dataset_descriptors = [] #list of numpy arrays. each array has shape = (num_img_keypoints, 128)

    for annotated_img in data:
        path = dataset_path + '/' + annotated_img['annotations'][0]['label'] + '/' + annotated_img['image']
       
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray_img, None)
        
        dataset_keypoints.append(kp)
        dataset_descriptors.append(des)

    with open(descriptors_path, 'wb') as f:
        pickle.dump(dataset_descriptors, f)

if __name__ == '__main__':
    main()