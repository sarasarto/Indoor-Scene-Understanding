import IPython.display
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
import utils_ade20k


# Load index with global information about ADE20K
DATASET_PATH = 'ADE20K_2021_17_01/'
index_file = 'index_ade20k.pkl'
with open('{}{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)
    print(type(index_ade20k))


#piccola parte di codice eseguita solo la prima 
#volta per visualizzare il file .pkl
#l'output è nel file "output.txt"
'''
with open('output.txt', 'a') as f:
    f.write(str(index_ade20k))
'''

examples = index_ade20k['filename']
training_examples = 0
validation_examples = 0
frame_examples = 0

for example in examples:
    if example.startswith('ADE_train'):
        training_examples += 1
    if example.startswith('ADE_val'):
        validation_examples += 1

    if example.startswith('ADE_frame'):
        frame_examples += 1

print('Training examples: ' + str(training_examples))
print('Validation examples: ' + str(validation_examples))
print('Frame examples: ' + str(frame_examples))
print('Total examples: ' + str(training_examples + validation_examples + frame_examples))
print('Folder lenght: ' + str(len(index_ade20k['folder'])))
print('Total scenes: ' + str(len(index_ade20k['scene'])))


with open('scenes.txt', 'a') as f:
    f.write(str(index_ade20k['scene']))


i = 5 # 16899, 16964
nfiles = len(index_ade20k['filename'])
file_name = index_ade20k['filename'][i]
num_obj = index_ade20k['objectPresence'][:, i].sum()
num_parts = index_ade20k['objectIsPart'][:, i].sum()
count_obj = index_ade20k['objectPresence'][:, i].max()
obj_id = np.where(index_ade20k['objectPresence'][:, i] == count_obj)[0][0]
obj_name = index_ade20k['objectnames'][obj_id]
full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])
print("The dataset has {} images".format(nfiles))
print("The image at index {} is {}".format(i, file_name))
print("It is located at {}".format(full_file_name))
print("It happens in a {}".format(index_ade20k['scene'][i]))
print("It has {} objects, of which {} are parts".format(num_obj, num_parts))
print("The most common object is object {} ({}), which appears {} times".format(obj_name, obj_id, count_obj))

root_path = DATASET_PATH
info = utils_ade20k.loadAde20K('{}'.format(full_file_name))
print(info)
img = cv2.imread(info['img_name'])[:,:,::-1]
seg = cv2.imread(info['segm_name'])[:,:,::-1]
seg_mask = seg.copy()

# The 0 index in seg_mask corresponds to background (not annotated) pixels
seg_mask[info['class_mask'] != obj_id+1] *= 0
plt.figure(figsize=(15,5))

plt.imshow(np.concatenate([img, seg, seg_mask], 1))
plt.axis('off')
if len(info['partclass_mask']):
    plt.figure(figsize=(5*len(info['partclass_mask']), 5))
    plt.title('Parts')
    plt.imshow(np.concatenate(info['partclass_mask'],1))
    plt.axis('off')

plt.show()