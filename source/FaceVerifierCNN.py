import torch.nn as nn
import torch

class FaceVerifierCNN(nn.Module):
    def __init__(self, input_size=64):
        super(FaceVerifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=0.15)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=0.2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(p=0.2)

        final_dim = input_size // 16
        self.flatten_size = 256 * final_dim * final_dim

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.drop4(x)
        x = x.view(-1, self.flatten_size)
        x = torch.relu(self.fc_bn(self.fc1(x)))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x  # logits