import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalClassifier(nn.Module):
    def __init__(self, dropout):
        super(ConvolutionalClassifier, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=8, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=8, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2)
        )
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x