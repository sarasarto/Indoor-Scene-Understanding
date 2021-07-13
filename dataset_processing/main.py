from dataset_info_manager import DatasetInfo
import json
import os


def main():
    dataset_info = DatasetInfo('ADE_20K/annotations')
    #info, mapping = dataset_info.get_dataset_info_and_obj_mapping()
    info, mapping = dataset_info.get_dataset_info_and_obj_mapping()

    with open('dataset_processing/dataset_info.json' ,'w') as f:
        json.dump(info, f)
    
    with open('dataset_processing/mapping.json' ,'w') as f:
        json.dump(mapping, f)

    info_all_objs, _ = dataset_info.get_dataset_info_and_obj_mapping(all_objects=True)
    
    with open('dataset_processing/dataset_info_all_objs.json' ,'w') as f:
        json.dump(info_all_objs, f)


if __name__ == '__main__':
    main()
