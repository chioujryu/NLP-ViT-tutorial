import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平輸入圖像
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x

# 設置超參數
input_size = 28*28  # MNIST 的圖像尺寸是 28x28
hidden_size = 128  # 隱藏層的神經元數量
num_classes = 10  # 分類類別數 (0~9)
num_epochs = 5  # 訓練輪數
batch_size = 64  # 批次大小
learning_rate = 0.001  # 學習率

# 準備 MNIST 數據集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 對圖像進行標準化
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化 MLP 模型
model = MLP(input_size, hidden_size, num_classes)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵損失函數
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用 Adam 優化器

# 訓練模型
def train_model():
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 將圖片和標籤移動到適當的設備（例如：GPU, 如果可用）
            images, labels = images, labels

            # 前向傳播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向傳播和優化
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 計算梯度
            optimizer.step()  # 更新權重

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 測試模型
def test_model():
    model.eval()  # 設置模型為評估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 在評估過程中不需要梯度
        for images, labels in test_loader:
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'測試集準確率：{100 * correct / total:.2f}%')

if __name__ == "__main__":
    train_model()
    test_model()
