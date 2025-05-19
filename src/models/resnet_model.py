import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50Classifier, self).__init__()

        # Load pretrained ResNet-50 model
        self.model = models.resnet50(pretrained=pretrained)

        # Modify the final classification layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
