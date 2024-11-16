import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)  # Adjust the size based on your image input
        self.fc2 = nn.Linear(512, 26)  # Output layer for 26 classes (alphabet)
        
        # New layers added between fc1 and fc2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(512)
    
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 256 * 28 * 28)  # Flatten the output for the fully connected layer
        x = self.relu(self.fc1(x))  # Apply ReLU activation on fc1 output
        x = self.batch_norm(x)  # Apply BatchNorm after fc1
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)  # Final output layer
        return x