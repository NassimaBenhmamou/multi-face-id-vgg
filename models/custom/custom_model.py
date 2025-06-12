# src/models/custom_model.py
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)  # on ne charge pas les poids ImageNet
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)
