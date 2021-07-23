import os
from shutil import copy
        

class DatasetConstructor():
    def __init__(self) -> None:
        self.new_path = 'dataset_ade20k_filtered'
        self.images_folder = 'images'
        self.masks_folder = 'masks'
        self.annotations_folder = 'annotations'
        self.root_paths = ['ADE20K_2021_17_01/images/ADE/training/home_or_hotel', 'ADE20K_2021_17_01/images/ADE/validation/home_or_hotel'] 
        self.folder_list = [self.images_folder, self.masks_folder, self.annotations_folder]

    def construct_filtered_dataset(self, scenes):

        for folder in self.folder_list:
            if not os.path.exists(os.path.join(self.new_path, folder)):
                os.makedirs(os.path.join(self.new_path, folder))

        for root_path in self.root_paths:
            for root, dirs, files in os.walk(root_path):
                scene = root.split('\\')[-1]
                if scene not in scenes:
                    continue
                for file_name in files:
                    if 'frame' in file_name:
                        continue
                    if file_name.endswith('.json'):
                        copy(os.path.join(root, file_name), os.path.join(self.new_path, self.annotations_folder))

                    elif file_name.endswith('.jpg'):
                        copy(os.path.join(root, file_name), os.path.join(self.new_path, self.images_folder))
                    
                    elif file_name.endswith('_seg.png'):
                        copy(os.path.join(root, file_name), os.path.join(self.new_path, self.masks_folder))