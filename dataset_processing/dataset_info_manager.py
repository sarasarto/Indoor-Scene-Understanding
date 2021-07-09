import os
import json

class DatasetInfo():
    def __init__(self, root_path='dataset_ade20k_filtered/annotations', scene_file='dataset_processing/scenes.txt', obj_classes_file='dataset_processing/interesting_objects.txt'):
        self.root_path = root_path
        self.scene_file = scene_file
        self.obj_classes_file = obj_classes_file
        self.scenes = []
        self.obj_classes = []
        
        with open(self.scene_file, 'r') as f:
            self.scenes = f.read().splitlines()
        with open(self.obj_classes_file, 'r') as f:
            self.obj_classes = f.read().splitlines()

    def get_dataset_info_and_obj_mapping(self): 
        info = {}
        mapping = {}
        scenes = {}
        instance_objs = {}
        dataset_size = 0
        mapped_values = 0

        for file in os.listdir(self.root_path):
            dataset_size += 1
            with open(os.path.join(self.root_path, file), 'r') as json_file:
                data = json.load(json_file)
                img_objs = data['annotation']['object']

                scene = data['annotation']['scene'][-1]
                if scene in self.scenes:
                    if scene not in scenes:
                        scenes[scene] = 1
                    else:
                        scenes[scene] += 1

            for obj in img_objs:
                if obj['raw_name'] in self.obj_classes:
                    if obj['raw_name'] not in mapping:
                        old_label = obj['name_ndx']
                        mapped_values += 1
                        new_label = mapped_values
                        mapping[obj['raw_name']] = {'old_label':old_label, 'new_label':new_label}

                    if obj['raw_name'] not in instance_objs:
                        instance_objs[obj['raw_name']] = 1
                    else:
                        instance_objs[obj['raw_name']] += 1
        
        info['dataset_size'] = dataset_size
        info['instances_per_obj'] = instance_objs
        info['instances_per_scene'] = scenes

        return info, mapping

    def get_dataset_info_and_ALL_OBJS(self): 
        info = {}
        mapping = {}
        scenes = {}
        instance_objs = {}
        dataset_size = 0
        mapped_values = 0

        for file in os.listdir(self.root_path):
            dataset_size += 1
            with open(os.path.join(self.root_path, file), 'r') as json_file:
                data = json.load(json_file)
                img_objs = data['annotation']['object']

                scene = data['annotation']['scene'][-1]
                if scene in self.scenes:
                    if scene not in scenes:
                        scenes[scene] = 1
                    else:
                        scenes[scene] += 1
                
                for obj in img_objs:
                    if scene in self.scenes:
                        if obj['raw_name'] not in mapping:
                            old_label = obj['name_ndx']
                            mapped_values += 1
                            new_label = mapped_values
                            mapping[obj['raw_name']] = {'old_label':old_label, 'new_label':new_label}

                        if obj['raw_name'] not in instance_objs:
                            instance_objs[obj['raw_name']] = 1
                        else:
                            instance_objs[obj['raw_name']] += 1
                    else:
                        break # se la scena non ci interessa non ha senso continuare
        
        info['dataset_size'] = dataset_size
        info['instances_per_obj'] = instance_objs
        info['instances_per_scene'] = scenes

        return info, mapping


