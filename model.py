import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)     # 16*3*3*1 + 16 = 160 params
        self.bn1 = nn.BatchNorm2d(16)                               # 16*2 = 32 params
        
        # Second conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)    # 32*16*3*3 + 32 = 4,640 params
        self.bn2 = nn.BatchNorm2d(32)                               # 32*2 = 64 params
        
        # Third conv block
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)    # 48*32*3*3 + 48 = 13,872 params
        self.bn3 = nn.BatchNorm2d(48)                               # 48*2 = 96 params
        
        # Pooling and fully connected
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 3 * 3, 10)                        # 48*3*3*10 + 10 = 4,330 params
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)      # 28x28 -> 14x14
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)      # 14x14 -> 7x7
        
        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)      # 7x7 -> 3x3
        
        # Classifier
        x = x.view(-1, 48 * 3 * 3)
        x = self.fc1(x)
        return x 