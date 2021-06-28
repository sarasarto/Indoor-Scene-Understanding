import os
import json

categories = []

with open('interesting_scenes.txt') as f:
    categories = f.read().splitlines()

#categories = ['living_room', 'kitchen'] questo era per provare con il dataset di prova

# train
data_dir = 'C:\\Users\\Cartella Giuseppe\\Desktop\\PROGETTO_CV\\ADE20K_2021_17_01\\images\\ADE\\'
f_img_train = open('C:\\Users\\Cartella Giuseppe\\Desktop\\PROGETTO_CV\\ADE20K_2021_17_01\\images\\ADE\\train_img.txt', 'w')
f_labels_train = open('C:\\Users\\Cartella Giuseppe\\Desktop\\PROGETTO_CV\\ADE20K_2021_17_01\\images\\ADE\\train_labels.txt', 'w')

f_img_test = open('C:\\Users\\Cartella Giuseppe\\Desktop\\PROGETTO_CV\\ADE20K_2021_17_01\\images\\ADE\\test_img.txt', 'w')
f_labels_test = open('C:\\Users\\Cartella Giuseppe\\Desktop\\PROGETTO_CV\\ADE20K_2021_17_01\\images\\ADE\\test_labels.txt', 'w')

with open('result.txt', 'r') as f:
    data = json.load(f)
    images = data['filename']
    scenes = data['scene']

for img, scene in zip(images, scenes):
    if img.startswith('ADE_train'):
        f_img_train.write(str(img) + "\n")
        f_labels_train.write(str(scene) + "\n")

    if img.startswith('ADE_val'):
        f_img_test.write(str(img) + "\n")
        f_labels_test.write(str(scene) + "\n")

print('salvataggio effettuato')