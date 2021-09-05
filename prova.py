import json

with open('ADE20K_filtering/filtered_dataset_info.json', 'rb') as f:
    data = json.load(f)

print(len(data['objects']))