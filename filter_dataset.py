import os
from shutil import copy

new_path = 'ADE_20K'
train_folder = 'training'
validation_folder = 'validation'
annotation_folder_training = 'training_annotations'
annotation_folder_validation = 'validation_annotations'
training_mask_folder = 'training_masks'
validation_mask_folder = 'validation_masks'

folder_list = [train_folder, validation_folder, annotation_folder_training, annotation_folder_validation, \
                training_mask_folder, validation_mask_folder]


for folder in folder_list:
    if not os.path.exists(os.path.join(new_path, folder)):
        os.makedirs(os.path.join(new_path, folder))


root_path = 'ADE20K_2021_17_01'
for root, dirs, files in os.walk(root_path):
    #print(root, dirs, files)
    for file_name in files:
        #print(file_name)
        if file_name.endswith('.json'):
            if 'train' in file_name:
                copy(os.path.join(root, file_name), os.path.join(new_path, annotation_folder_training))
                
            if 'val' in file_name:
                copy(os.path.join(root, file_name), os.path.join(new_path, annotation_folder_validation))

        elif file_name.endswith('.jpg'):
            if 'train' in file_name:
                copy(os.path.join(root, file_name), os.path.join(new_path, train_folder))
                
            if 'val' in file_name:
                copy(os.path.join(root, file_name), os.path.join(new_path, validation_folder))
        
        elif file_name.endswith('_seg.png'):
            if 'train' in file_name:
                copy(os.path.join(root, file_name), os.path.join(new_path, training_mask_folder))
                
            if 'val' in file_name:
                copy(os.path.join(root, file_name), os.path.join(new_path, validation_mask_folder))

    
        