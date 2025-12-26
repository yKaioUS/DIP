import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from CNN import CNN

# 自定义数据集
class CharImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_names = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # 灰度图
        if self.transform:
            image = self.transform(image)
        return image, label

def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch [{epoch + 1} / {epochs}], step {i + 1} / {len(train_loader)}, Loss: {loss.item()}")

def test_model(model, dataset):  # 新增测试函数
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for class_idx in range(len(dataset.class_names)):
            class_name = dataset.class_names[class_idx]
            class_dir = os.path.join(dataset.root_dir, class_name)
            files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
            files = files[:10]  # 只取前10个
            for fname in files:
                img_path = os.path.join(class_dir, fname)
                image = Image.open(img_path).convert('L')
                if dataset.transform:
                    image = dataset.transform(image)
                image = image.unsqueeze(0).to(device)
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                if predicted.item() == class_idx:
                    correct += 1
    print(f"测试集前10张/类的准确率: {100 * correct / total:.2f}% ({correct}/{total})")

# 定义超参数
num_epochs = 5  # 训练轮数
batch_size = 16  # 适当减小batch_size有助于模型泛化
learning_rate = 0.001  # 降低学习率让模型收敛更细致

# 数据增强和归一化
transform = transforms.Compose([transforms.ToTensor()])

# 加载和预处理数据
train_dataset = CharImageDataset(root_dir='./cnn_char_train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.class_names)
model = CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
train(model, train_loader, optimizer, criterion, epochs=num_epochs)

# 保存模型参数到文件夹
save_dir = './'  # 你想保存的文件夹路径
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'char_cnn.pth')
torch.save(model.state_dict(), model_path)
print(f"模型参数已保存到: {model_path}")

test_model(model, train_dataset)