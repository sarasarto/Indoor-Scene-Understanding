import os
import json

path = 'ADE_20K/training_annotations'

objects = set()

for f in os.listdir(path):
    with open(os.path.join(path, f), 'r') as f:
        data = json.load(f)
        obj_list = data['annotation']['object']

        for obj in obj_list:
            objects.add(obj['raw_name'])


with open('interesting_objects.txt', 'a') as f:
    for obj in objects:
        f.write(obj + '\n')




