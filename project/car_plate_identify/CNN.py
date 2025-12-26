import torch.nn as nn
number = 44
# 定义CNN网络
class CNN(nn.Module):
    def __init__(self, num_classes=number):  # 有number类
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=6, kernel_size=5, padding=2) # 28*28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 14*14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 10*10
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 5*5
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.reshape(-1, 16*5*5)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
