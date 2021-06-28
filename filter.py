import pickle as pkl
import utils_ade20k
import copy
import numpy as np
import json


classes = []

with open('interesting_scenes.txt') as f:
    classes = f.read().splitlines()

DATASET_PATH = ''
index_file = 'index_ade20k.pkl'
with open('{}{}'.format(DATASET_PATH, index_file), 'rb') as f:
    pkl_file = pkl.load(f)
    index_ade20k = copy.deepcopy(pkl_file)

print(type(index_ade20k['objectIsPart']))
print(len(index_ade20k['objectnames']))

inter_examples = 0

'''
* filename: array of length N=27574 with the image file names
* folder: array of length N with the image folder names.
* scene: array of length N providing the scene name (same classes as the Places database) for each image.
* objectIsPart: array of size [C, N] counting how many times an object is a part in each image. objectIsPart[c,i]=m if in image i object class c is a part of another object m times. For objects, objectIsPart[c,i]=0, and for parts we will find: objectIsPart[c,i] = objectPresence(c,i)
* objectPresence: array of size [C, N] with the object counts per image. objectPresence(c,i)=n if in image i there are n instances of object class c.
* objectcounts: array of length C with the number of instances for each object class.
* objectnames: array of length C with the object class names.
* proportionClassIsPart: array of length C with the proportion of times that class c behaves as a part. If proportionClassIsPart[c]=0 then it means that this is a main object (e.g., car, chair, ...). See bellow for a discussion on the utility of this variable.
* wordnet_found: array of length C. It indicates if the objectname was found in Wordnet.
* wordnet_level1: list of length C. WordNet associated.
* wordnet_synset: list of length C. WordNet synset for each object name. Shows the full hierarchy separated by .
* wordnet_hypernym: list of length C. WordNet hypernyms for each object name.
* wordnet_gloss: list of length C. WordNet definition.
* wordnet_synonyms: list of length C. Synonyms for the WordNet definition.
* wordnet_frequency: array of length C. How many times each wordnet appears
'''

#inizio a preparare il nuovo dizionario
new_dict = {'filename': [],
            'folder': [],
            'scene': [],
            #'objectIsPart': np.array([]),
            #'objectPresence': np.array([]),
            #objectcounts non ci serve
            #'objectnames': index_ade20k['objectnames'],
            #'proportionClassIsPart': index_ade20k['proportionClassIsPart'],
            #'wordnet_found': index_ade20k['wordnet_found'],
            #'wordnet_level1': index_ade20k['wordnet_level1'],
            #'wordnet_synset': index_ade20k['wordnet_synset'],
            #'wordnet_hypernym': index_ade20k['wordnet_hypernym'],
            #'wordnet_gloss': index_ade20k['wordnet_gloss'],
            #'wordnet_synonyms': index_ade20k['wordnet_synonyms'],
            #'wordnet_frequency': index_ade20k['wordnet_frequency']
            }


idx_to_keep = []
scenes = pkl_file['scene']
for i, scene in enumerate(scenes):
    scene = scene.split('/')

    if 'bakery' in scene or 'outdoor' in scene or 'exterior' in scene:
        continue
    intersection = list(set(classes) & set(scene))

    if len(intersection) > 0:
        new_dict['filename'].append(index_ade20k['filename'][i])
        new_dict['folder'].append(index_ade20k['folder'][i])

        #cleaning the scene
        scene = index_ade20k['scene'][i]
        scene = scene.strip('/').split('/')[0]
        new_dict['scene'].append(scene)

        idx_to_keep.append(i)
        inter_examples += 1
        

with open('result.txt', 'w') as f:
        json.dump(new_dict, f)


new_dict['objectIsPart'] = index_ade20k['objectIsPart'][:, idx_to_keep]
new_dict['objectPresence'] = index_ade20k['objectPresence'][:, idx_to_keep]

print('Num scene utili: ' + str(inter_examples))

with open('results.txt', 'a') as f:
    f.write(str(index_ade20k))

with open('scenes.txt', 'a') as f:
    f.write(str(index_ade20k['scene']))

    



    