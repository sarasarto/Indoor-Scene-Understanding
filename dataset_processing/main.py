from dataset_info_manager import DatasetInfo
import json
import os


def main():
    dataset_info = DatasetInfo('ADE_20K/annotations')
    info, mapping = dataset_info.get_dataset_info_and_obj_mapping()

    with open('data_processing/dataset_info.json' ,'w') as f:
        json.dump(info, f)

    with open('data_processing/mapping.json' ,'w') as f:
        json.dump(mapping, f)


if __name__ == '__main__':
    main()
