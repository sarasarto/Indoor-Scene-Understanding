# Indoor-Scene-Understanding
Authors: Giuseppe Cartella, Sara Sarto, Kevin Marchesini
Computer Vision project for the analysis of indoor house scenes including a furniture retrieval system.

Preliminary steps
Download from drive (request access to authors) the folder containing all trained models and dataset.

Copy all dataset folders to /retrieval folder of the project.

Move 'model_mask_default.pt' and 'model_mask_modified.pt' to /Indoor-Scene-Understanding

Move 'dataset_embedding.pt' to /Indoor-Scene-Understanding/retrieval/method_autoencoder

Move 'descriptors.pkl' to /Indoor-Scene-Understanding/retrieval/method_SIFT

Move 'MLP_model.pt' and 'dataset_all_objects.csv' to /Indoor-Scene-Understanding/classification

Install the requirements.txt in your virtual environment 
pip install -r requirements.txt


Execute the pipeline
In the root folder execute python execute_pipeline.py -img test_images/bedroom.jpg -mdl modified -rtv autoencoder -clf forest
