import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

class HomeScenesClassifier(nn.Module):

    def __init__(self, num_objs, num_classes: int = 11):
        super(HomeScenesClassifier, self).__init__()

        self.fully = nn.Sequential(
            nn.Linear(num_objs , 4096),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fully(x)
        return x