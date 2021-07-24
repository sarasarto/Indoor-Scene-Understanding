from .training_utils import get_instance_model_default, get_instance_model_modified
import torch
import numpy as np

class PredictionModel():
    def __init__(self, model_path, num_classes, default_model=True,) -> None:
        self.model_path = model_path
        self.model = None
        self.prediction = None

        if default_model:
            self.model = get_instance_model_default(num_classes)
        else:
            self.model = get_instance_model_modified(num_classes)
        
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
        self.model.eval()

    def segment_image(self, img):
        self.prediction = self.model([img])
        return self.prediction
    
    def extract_furniture(self, prediction, score_threshold=0.7):
        if self.prediction is None:
            raise ValueError('Error: first you must segment the image!')
        
        scores = prediction[0]['scores']
        scores = scores[scores > score_threshold]
        num_objs = len(scores)
        labels = prediction[0]['labels'][:num_objs].detach().numpy()
        boxes = prediction[0]['boxes'][:num_objs,:].detach().numpy()
        masks = prediction[0]['masks'][:num_objs,:,:].detach().numpy().round()
        if len(masks) > 1:
            masks = np.squeeze(masks)
        else:
            masks = np.squeeze(masks, axis=0)

        return boxes, masks, labels, scores
       



