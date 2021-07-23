from ade20k_utils import DatasetExplorer
from construct_filtered_dataset import DatasetConstructor
import json
import os

def main():
    #first we extract info about original complete dataset
    interesting_scenes = []
    with open('ADE20K_filtering/interesting_scenes.txt') as f:
        interesting_scenes = f.read().splitlines()

    pkl_file = 'ADE20K_2021_17_01/index_ade20k.pkl'
    original_dataset_info = 'ADE20K_filtering/original_dataset_info.json'
    filtered_dataset_info = 'ADE20K_filtering/filtered_dataset_info.json'
    filtered_dataset = 'dataset_ade20k_filtered'
    dxl = DatasetExplorer('ADE20K_filtering')

    if not os.path.exists(original_dataset_info):
        dxl.original_dataset_info(pkl_file)

    #now we construct the filtered dataset
    if not os.path.exists(filtered_dataset):
        dc = DatasetConstructor()
        dc.construct_filtered_dataset(interesting_scenes)
    
    #extract info from filtered dataset
    if not os.path.exists(filtered_dataset_info):
        dxl.filtered_dataset_info(filtered_dataset + '/annotations')

    #check invalid bboxes
    if not os.path.exists('ADE20K_filtering/invalid_masks.json'):
        invalid_mask_imgs = dxl.check_invalid_bboxes(filtered_dataset + '/masks')
    else:
        with open('ADE20K_filtering/invalid_masks.json') as f:
            invalid_mask_imgs = json.load(f)
    
    dxl.delete_faulty_examples(filtered_dataset, invalid_mask_imgs['invalid_masks'])

if __name__ == '__main__':
    main()
