# Indoor-Scene-Understanding
## Authors: Giuseppe Cartella, Sara Sarto, Kevin Marchesini
Computer Vision project for the analysis of indoor house scenes including a furniture retrieval system.

## Preliminary steps
1. Download from drive (request access to authors) the folder containing all trained models and dataset.

2. Copy all dataset folders to /retrieval folder of the project.

3. Move 'model_mask_default.pt' and 'model_mask_modified.pt' to /Indoor-Scene-Understanding

4. Move 'dataset_embedding.pt' to /Indoor-Scene-Understanding/retrieval/method_autoencoder

5. Move 'descriptors.pkl' to /Indoor-Scene-Understanding/retrieval/method_SIFT

6. Move 'MLP_model.pt' and 'dataset_all_objects.csv' to /Indoor-Scene-Understanding/classification

7. Install the requirements.txt in your virtual environment 
8. 
  '''
  pip install -r requirements.txt
  '''


## Execute the pipeline
In the root folder execute:
Python execute_pipeline.py -img test_images/bedroom.jpg -mdl modified -rtv autoencoder -clf forest
