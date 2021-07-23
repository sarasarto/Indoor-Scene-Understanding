import pickle as pkl
import json
import os
from PIL import Image
import numpy as np

class DatasetExplorer():
    def __init__(self, dataset_stats_root) -> None:
        self.dataset_stats_root = dataset_stats_root

    def original_dataset_info(self, pklfile_path, out_json_file='original_dataset_info.json'):
        with open(pklfile_path, 'rb') as f:
            pkl_file = pkl.load(f)
        
        #overall dataset info(including not interesting scenes)
        images = pkl_file['filename']
        num_images = len(images) #include train, test, frame
        scenes = np.array(pkl_file['scene'])
        unique_scenes, counts = np.unique(scenes, return_counts=True)
        counts = counts.tolist()
        instances_for_scene = dict(zip(unique_scenes, counts))

        num_train_images = 0
        num_val_images = 0
        num_frame_images = 0
        for file in images:
            if 'train' in file:
                num_train_images += 1
            if 'val' in file:
                num_val_images += 1
            if 'frame' in file:
                num_frame_images += 1
        
        with open(os.path.join(self.dataset_stats_root,out_json_file), 'w') as f:
            data = {'num_images':num_images, 'num_train_images':num_train_images,
                    'num_val_images':num_val_images, 'num_frame_images':num_frame_images,
                    'instances_for_scenes':instances_for_scene}
            json.dump(data, f)
    
    def filtered_dataset_info(self, annotations_path, out_json_file='filtered_dataset_info.json'):
        num_images = 0
        scenes = {}
        objects = {}
        total_objs = 0
        for file in os.listdir(annotations_path):
            num_images += 1
            with open(os.path.join(annotations_path, file), 'r') as json_file:
                data = json.load(json_file)
                img_objs = data['annotation']['object']

                #update number of scenes
                scene = data['annotation']['scene'][-1]
                if scene not in scenes:
                    scenes[scene] = 1
                else:
                    scenes[scene] += 1

            #consider all objects
            for obj in img_objs:
                numeric_label = obj['name_ndx']
                text_label = obj['raw_name']

                if numeric_label not in objects:
                    numeric_label = int(numeric_label)
                    total_objs += 1 #update here because different text label
                                    #can have same numeric label
                    objects[numeric_label] = {'num_instances': 1, 
                                              'labels':[text_label], 
                                              'new_label':total_objs} #define new label for training between [1,n]
                else:
                    objects[numeric_label]['num_instances'] += 1
                    if text_label not in objects[numeric_label]['labels']:
                        objects[numeric_label]['labels'].append(text_label)

        with open(os.path.join(self.dataset_stats_root,out_json_file), 'w') as f:
            data = {'num_images':num_images,
                    'objects':objects, 
                    'instances_for_scenes':scenes}
            json.dump(data, f)


    def check_invalid_bboxes(self, masks_root_path):
        count = 0
        invalid_mask_imgs = [] #there are some faulty bounding boxes
        for file_name in os.listdir(masks_root_path): 
            count += 1
            if count%10==0:
                print(count)
            mask_path = os.path.join(masks_root_path, file_name)
            mask = Image.open(mask_path)
            # convert the PIL Image into a numpy array
            mask = np.array(mask)

            # instances are encoded as different colors
            obj_ids = np.unique(mask[:,:,2]) #Ade saves instances on B channel of mask
            # first id is the background, so remove it
            obj_ids = obj_ids[1:] 
            
            # split the color-encoded mask into a set
            # of binary masks
            masks = mask[:,:,2] == obj_ids[:, None, None]

            for i in range(len(masks)):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmin == xmax or ymin == ymax:
                    #raise ValueError('Invalid bounding box!')
                    print(f'Founded invalid bounding box in mask: {file_name}!')
                    invalid_mask_imgs.append(file_name)
                    break

        with open('ADE20K_filtering/invalid_masks.json', 'w') as f:
            invalid_masks = {'invalid_masks':invalid_mask_imgs}
            json.dump(invalid_masks, f)
        return invalid_masks

    #faulty example=mask with an invalid bbox
    def delete_faulty_examples(self, root_path, invalid_masks):
        mask_path = os.path.join(root_path, 'masks')
        images_path = os.path.join(root_path, 'images')
        annotations_path = os.path.join(root_path, 'annotations')
        for i in invalid_masks: 
            mask_file = i
            img_file = i.split('_seg.png')[0] + '.jpg'
            annotation_file = i.split('_seg.png')[0] + '.json'

            if os.path.exists(os.path.join(mask_path, mask_file)):
                os.remove(os.path.join(mask_path, mask_file))
            if os.path.exists(os.path.join(images_path, img_file)):
                os.remove(os.path.join(images_path, img_file))
            if os.path.exists(os.path.join(annotations_path, annotation_file)):
                os.remove(os.path.join(annotations_path, annotation_file))