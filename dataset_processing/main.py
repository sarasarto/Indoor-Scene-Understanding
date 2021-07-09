from dataset_info_manager import DatasetInfo
import json
import os


def main():
    dataset_info = DatasetInfo('ADE_20K/annotations')
    #info, mapping = dataset_info.get_dataset_info_and_obj_mapping()
    info_all, mapping_all = dataset_info.get_dataset_info_and_ALL_OBJS()

    #with open('dataset_processing/dataset_info.json' ,'w') as f:
    #    json.dump(info, f)

    #with open('dataset_processing/mapping.json' ,'w') as f:
    #    json.dump(mapping, f)

    with open('dataset_processing/ALL_OBJs_dataset_info.json' ,'w') as f:
        json.dump(info_all, f)


if __name__ == '__main__':
    main()
