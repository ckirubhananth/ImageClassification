import torch.nn as nn
import torchvision.models as models

class EfficientNetB2Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB2Classifier, self).__init__()

        # Load pretrained EfficientNet-B2 model
        self.model = models.efficientnet_b2(pretrained=pretrained)

        # Modify the final classification layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
