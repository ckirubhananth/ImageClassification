import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """ Convolutional Neural Network for image classification. """
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        # **Dynamically compute fc1 input size instead of hardcoding**
        self._fc_input_size = self._compute_fc_input_size()

        self.fc1 = nn.Linear(self._fc_input_size, 128)  # Use computed size
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def _compute_fc_input_size(self):
        """ Compute correct input size for fc1 dynamically. """
        with torch.no_grad():
            x = torch.zeros((1, 3, 64, 64))  # Simulate input with expected dimensions
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.pool2(x)

            return x.view(x.size(0), -1).shape[1]  # Calculate size for fc1 dynamically

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
