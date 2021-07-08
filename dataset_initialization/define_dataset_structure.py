import os
from shutil import copy

new_path = 'dataset_ade20k_filtered'
images_folder = 'training'
masks_folder = 'masks'
annotations_folder = 'annotations'

folder_list = [images_folder, masks_folder, annotations_folder]


for folder in folder_list:
    if not os.path.exists(os.path.join(new_path, folder)):
        os.makedirs(os.path.join(new_path, folder))

root_path = 'ADE20K_2021_17_01'

for root, dirs, files in os.walk(root_path):
    #print(root, dirs, files)
    for file_name in files:
        #print(file_name)
        if file_name.endswith('.json'):
            copy(os.path.join(root, file_name), os.path.join(new_path, annotations_folder))

        elif file_name.endswith('.jpg'):
            copy(os.path.join(root, file_name), os.path.join(new_path, images_folder))
        
        elif file_name.endswith('_seg.png'):
           copy(os.path.join(root, file_name), os.path.join(new_path, masks_folder))
    
        